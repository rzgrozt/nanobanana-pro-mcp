"""Nano Banana Pro MCP Server - Image generation, conversion, and smart asset placement."""

import argparse
import asyncio
import base64
import os
import re
import tempfile
from pathlib import Path

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent
from openai import OpenAI
from PIL import Image

# Global client and config
_client: OpenAI | None = None
_model: str = "gemini-3-pro-image"


def get_client() -> OpenAI:
    """Get the configured OpenAI client."""
    if _client is None:
        raise RuntimeError("Client not initialized. Ensure --api-key and --base-url are provided.")
    return _client


# Aspect ratio detection keywords
ASPECT_RATIOS = {
    "1:1": {
        "size": "1024x1024",
        "keywords": ["icon", "avatar", "logo", "profile", "thumbnail", "square", "favicon", "badge", "emoji"]
    },
    "16:9": {
        "size": "1280x720",
        "keywords": ["banner", "hero", "header", "cover", "landscape", "wide", "youtube", "video", "thumbnail"]
    },
    "9:16": {
        "size": "720x1280",
        "keywords": ["mobile", "story", "stories", "splash", "portrait", "vertical", "phone", "instagram"]
    },
    "4:3": {
        "size": "1216x896",
        "keywords": ["photo", "traditional", "classic", "standard"]
    }
}


def detect_aspect_ratio(prompt: str) -> str:
    """Detect the best aspect ratio based on prompt keywords."""
    prompt_lower = prompt.lower()

    for ratio, config in ASPECT_RATIOS.items():
        for keyword in config["keywords"]:
            if keyword in prompt_lower:
                return config["size"]

    # Default to square
    return "1024x1024"


def extract_filename_from_prompt(prompt: str) -> str:
    """Extract a clean filename from the generation prompt."""
    # Remove common filler words
    stopwords = {"a", "an", "the", "of", "for", "with", "in", "on", "at", "to", "and", "or", "draw", "create", "generate", "make", "design"}

    # Clean and split
    words = re.sub(r'[^\w\s-]', '', prompt.lower()).split()
    words = [w for w in words if w not in stopwords and len(w) > 1]

    # Take first 4 meaningful words
    filename_words = words[:4]

    if not filename_words:
        filename_words = ["image"]

    return "-".join(filename_words)


# Project type detection patterns
PROJECT_PATTERNS = {
    "nextjs": {
        "markers": ["next.config.js", "next.config.mjs", "next.config.ts", ".next"],
        "image_dirs": ["public/images", "public/assets", "public"],
        "naming_pattern": "kebab-case"
    },
    "react": {
        "markers": ["src/App.tsx", "src/App.jsx", "src/index.tsx", "src/index.jsx"],
        "image_dirs": ["src/assets/images", "src/assets", "public/images", "public"],
        "naming_pattern": "kebab-case"
    },
    "vue": {
        "markers": ["vue.config.js", "vite.config.ts", "src/App.vue"],
        "image_dirs": ["src/assets/images", "src/assets", "public/images", "public"],
        "naming_pattern": "kebab-case"
    },
    "python": {
        "markers": ["pyproject.toml", "setup.py", "requirements.txt"],
        "image_dirs": ["assets", "images", "static/images", "resources"],
        "naming_pattern": "snake_case"
    },
    "unity": {
        "markers": ["Assets", "ProjectSettings", "*.unity"],
        "image_dirs": ["Assets/Images", "Assets/Textures", "Assets/Sprites"],
        "naming_pattern": "PascalCase"
    },
    "godot": {
        "markers": ["project.godot", "*.godot"],
        "image_dirs": ["assets/images", "assets/sprites", "resources/images"],
        "naming_pattern": "snake_case"
    },
    "electron": {
        "markers": ["electron.js", "main.js", "electron-builder.json"],
        "image_dirs": ["assets", "resources", "public/images"],
        "naming_pattern": "kebab-case"
    },
    "generic": {
        "markers": [],
        "image_dirs": ["images", "assets", "img", "."],
        "naming_pattern": "kebab-case"
    }
}


def detect_project_type(project_path: Path) -> str:
    """Detect the project type based on marker files."""
    for project_type, config in PROJECT_PATTERNS.items():
        if project_type == "generic":
            continue
        for marker in config["markers"]:
            if "*" in marker:
                # Glob pattern
                if list(project_path.glob(marker)):
                    return project_type
            else:
                if (project_path / marker).exists():
                    return project_type
    return "generic"


def apply_naming_convention(name: str, convention: str) -> str:
    """Apply a naming convention to a filename."""
    # First normalize to words
    words = re.split(r'[-_\s]+', name.lower())

    if convention == "kebab-case":
        return "-".join(words)
    elif convention == "snake_case":
        return "_".join(words)
    elif convention == "PascalCase":
        return "".join(word.capitalize() for word in words)
    elif convention == "camelCase":
        return words[0] + "".join(word.capitalize() for word in words[1:])
    else:
        return "-".join(words)


def detect_existing_naming_convention(directory: Path) -> str | None:
    """Detect the naming convention used in existing files."""
    if not directory.exists():
        return None

    image_extensions = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".ico", ".svg"}
    image_files = [f for f in directory.iterdir() if f.suffix.lower() in image_extensions]

    if not image_files:
        return None

    # Analyze filenames
    kebab_count = sum(1 for f in image_files if "-" in f.stem and "_" not in f.stem)
    snake_count = sum(1 for f in image_files if "_" in f.stem and "-" not in f.stem)
    pascal_count = sum(1 for f in image_files if f.stem[0].isupper() and "-" not in f.stem and "_" not in f.stem)

    counts = {"kebab-case": kebab_count, "snake_case": snake_count, "PascalCase": pascal_count}
    best = max(counts, key=counts.get)

    if counts[best] > 0:
        return best
    return None


def find_best_image_directory(project_path: Path, project_type: str) -> Path:
    """Find or create the best directory for images."""
    config = PROJECT_PATTERNS.get(project_type, PROJECT_PATTERNS["generic"])

    # Check existing directories
    for dir_path in config["image_dirs"]:
        full_path = project_path / dir_path
        if full_path.exists() and full_path.is_dir():
            return full_path

    # Create the first preferred directory
    preferred = project_path / config["image_dirs"][0]
    preferred.mkdir(parents=True, exist_ok=True)
    return preferred


# ICO sizes for Windows icons
ICO_SIZES = [(16, 16), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)]

# ICNS sizes for macOS icons
ICNS_SIZES = [(16, 16), (32, 32), (64, 64), (128, 128), (256, 256), (512, 512), (1024, 1024)]


def convert_to_ico(source: Path, dest: Path) -> None:
    """Convert an image to ICO format with multiple sizes."""
    img = Image.open(source)

    # Ensure RGBA
    if img.mode != "RGBA":
        img = img.convert("RGBA")

    # Create resized versions
    icons = []
    for size in ICO_SIZES:
        resized = img.resize(size, Image.Resampling.LANCZOS)
        icons.append(resized)

    # Save as ICO
    icons[0].save(dest, format="ICO", sizes=[s for s in ICO_SIZES])


def convert_to_icns(source: Path, dest: Path) -> None:
    """Convert an image to ICNS format (macOS)."""
    img = Image.open(source)

    # Ensure RGBA
    if img.mode != "RGBA":
        img = img.convert("RGBA")

    # For ICNS, we need to save as PNG first, then use iconutil on macOS
    # Since we can't guarantee macOS, we'll save the largest size as PNG
    # and note that full ICNS requires macOS tooling
    largest = max(ICNS_SIZES, key=lambda x: x[0])
    resized = img.resize(largest, Image.Resampling.LANCZOS)

    # Save as ICNS (Pillow has basic ICNS support)
    resized.save(dest, format="ICNS")


def convert_image_format(source: Path, target_format: str, dest: Path | None = None) -> Path:
    """Convert an image to the specified format."""
    img = Image.open(source)

    # Determine destination path
    if dest is None:
        dest = source.with_suffix(f".{target_format.lower()}")

    target_format = target_format.lower()

    if target_format == "ico":
        convert_to_ico(source, dest)
    elif target_format == "icns":
        convert_to_icns(source, dest)
    elif target_format in ("jpg", "jpeg"):
        # JPEG doesn't support alpha
        if img.mode in ("RGBA", "P"):
            img = img.convert("RGB")
        img.save(dest, format="JPEG", quality=95)
    elif target_format == "webp":
        img.save(dest, format="WEBP", quality=95)
    elif target_format == "png":
        img.save(dest, format="PNG")
    else:
        raise ValueError(f"Unsupported format: {target_format}")

    return dest


# Create MCP server
server = Server("nanobanana-pro-mcp")


@server.list_tools()
async def list_tools() -> list[Tool]:
    """List available tools."""
    return [
        Tool(
            name="generate_image",
            description="Generate an image using AI based on a text prompt. Automatically detects the best aspect ratio based on the prompt context (e.g., 'icon' -> square, 'banner' -> wide). Returns a temporary file path and suggested filename.",
            inputSchema={
                "type": "object",
                "properties": {
                    "prompt": {
                        "type": "string",
                        "description": "Text description of the image to generate"
                    }
                },
                "required": ["prompt"]
            }
        ),
        Tool(
            name="convert_image",
            description="Convert an image file to a different format. Supports: png, jpg, webp, ico (Windows icon with multiple sizes), icns (macOS icon). For ico/icns, automatically generates all required size variants.",
            inputSchema={
                "type": "object",
                "properties": {
                    "source_path": {
                        "type": "string",
                        "description": "Path to the source image file"
                    },
                    "format": {
                        "type": "string",
                        "enum": ["png", "jpg", "webp", "ico", "icns"],
                        "description": "Target format to convert to"
                    }
                },
                "required": ["source_path", "format"]
            }
        ),
        Tool(
            name="save_asset",
            description="Save an image to the appropriate project location with smart naming. Automatically detects project type (Next.js, React, Vue, Unity, Godot, etc.), finds the best directory for images, and applies the project's naming convention.",
            inputSchema={
                "type": "object",
                "properties": {
                    "source_path": {
                        "type": "string",
                        "description": "Path to the image file to save"
                    },
                    "project_path": {
                        "type": "string",
                        "description": "Root path of the project"
                    },
                    "suggested_name": {
                        "type": "string",
                        "description": "Optional suggested name for the file (without extension)"
                    }
                },
                "required": ["source_path", "project_path"]
            }
        )
    ]


@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    """Handle tool calls."""

    if name == "generate_image":
        prompt = arguments["prompt"]

        # Detect aspect ratio
        size = detect_aspect_ratio(prompt)

        # Generate image
        client = get_client()
        response = client.chat.completions.create(
            model=_model,
            extra_body={"size": size},
            messages=[{"role": "user", "content": prompt}]
        )

        # Get the response content (base64 image or URL)
        content = response.choices[0].message.content

        # Create temp file
        suggested_name = extract_filename_from_prompt(prompt)
        temp_dir = tempfile.gettempdir()
        temp_path = Path(temp_dir) / f"{suggested_name}.png"

        # Handle response - could be base64 or URL
        if content.startswith("data:image"):
            # Base64 data URL
            header, data = content.split(",", 1)
            image_data = base64.b64decode(data)
            temp_path.write_bytes(image_data)
        elif content.startswith("http"):
            # URL - download it
            import urllib.request
            urllib.request.urlretrieve(content, temp_path)
        else:
            # Assume raw base64
            try:
                image_data = base64.b64decode(content)
                temp_path.write_bytes(image_data)
            except Exception:
                # Maybe it's a file path or error message
                return [TextContent(
                    type="text",
                    text=f"Unexpected response from image API: {content[:200]}..."
                )]

        return [TextContent(
            type="text",
            text=f"Image generated successfully!\n\nTemporary path: {temp_path}\nSuggested name: {suggested_name}\nSize: {size}\n\nUse save_asset to place it in your project, or convert_image to change the format."
        )]

    elif name == "convert_image":
        source_path = Path(arguments["source_path"])
        target_format = arguments["format"]

        if not source_path.exists():
            return [TextContent(type="text", text=f"Error: Source file not found: {source_path}")]

        try:
            dest_path = convert_image_format(source_path, target_format)

            extra_info = ""
            if target_format == "ico":
                extra_info = f"\nIncluded sizes: {', '.join(f'{w}x{h}' for w, h in ICO_SIZES)}"
            elif target_format == "icns":
                extra_info = "\nNote: Full ICNS support requires macOS. Basic conversion applied."

            return [TextContent(
                type="text",
                text=f"Image converted successfully!\n\nOutput: {dest_path}\nFormat: {target_format.upper()}{extra_info}"
            )]
        except Exception as e:
            return [TextContent(type="text", text=f"Error converting image: {e}")]

    elif name == "save_asset":
        source_path = Path(arguments["source_path"])
        project_path = Path(arguments["project_path"])
        suggested_name = arguments.get("suggested_name")

        if not source_path.exists():
            return [TextContent(type="text", text=f"Error: Source file not found: {source_path}")]

        if not project_path.exists():
            return [TextContent(type="text", text=f"Error: Project path not found: {project_path}")]

        # Detect project type
        project_type = detect_project_type(project_path)

        # Find best directory
        image_dir = find_best_image_directory(project_path, project_type)

        # Determine naming convention
        existing_convention = detect_existing_naming_convention(image_dir)
        if existing_convention:
            convention = existing_convention
        else:
            convention = PROJECT_PATTERNS[project_type]["naming_pattern"]

        # Determine filename
        if suggested_name:
            base_name = suggested_name
        else:
            base_name = source_path.stem

        # Apply convention
        final_name = apply_naming_convention(base_name, convention)

        # Keep original extension
        extension = source_path.suffix

        # Handle duplicates
        dest_path = image_dir / f"{final_name}{extension}"
        counter = 1
        while dest_path.exists():
            dest_path = image_dir / f"{final_name}-{counter}{extension}"
            counter += 1

        # Copy file
        import shutil
        shutil.copy2(source_path, dest_path)

        return [TextContent(
            type="text",
            text=f"Asset saved successfully!\n\nPath: {dest_path}\nProject type: {project_type}\nNaming convention: {convention}\nDirectory: {image_dir}"
        )]

    else:
        return [TextContent(type="text", text=f"Unknown tool: {name}")]


def main():
    """Main entry point."""
    global _client, _model

    parser = argparse.ArgumentParser(description="Nano Banana Pro MCP Server")
    parser.add_argument("--api-key", required=True, help="API key for the image generation service")
    parser.add_argument("--base-url", required=True, help="Base URL for the image generation API")
    parser.add_argument("--model", default="gemini-3-pro-image", help="Model to use for image generation")

    args = parser.parse_args()

    # Initialize client
    _client = OpenAI(
        api_key=args.api_key,
        base_url=args.base_url
    )
    _model = args.model

    # Run server
    async def run():
        async with stdio_server() as (read_stream, write_stream):
            await server.run(read_stream, write_stream, server.create_initialization_options())

    asyncio.run(run())


if __name__ == "__main__":
    main()
