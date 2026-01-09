# Nano Banana Pro Image Generation for Claude Code

An MCP (Model Context Protocol) server that provides AI-powered image generation, format conversion, and smart asset placement for Claude Code.

## Features

- **`generate_image`** - Generate images from text prompts with automatic aspect ratio detection
- **`convert_image`** - Convert images between formats (PNG, JPG, WebP, ICO, ICNS)
- **`save_asset`** - Smart asset placement with project-aware naming conventions

### Smart Detection

- **Aspect Ratio**: Automatically detects the best size based on prompt keywords
  - "icon", "avatar", "logo" → 1:1 (1024x1024)
  - "banner", "hero", "header" → 16:9 (1280x720)
  - "mobile", "story", "splash" → 9:16 (720x1280)
  - "photo", "traditional" → 4:3 (1216x896)

- **Project Type**: Detects your project framework and places assets accordingly
  - Next.js, React, Vue → `public/images/` or `src/assets/`
  - Unity → `Assets/Images/`
  - Godot → `assets/images/`
  - Python → `assets/` or `static/images/`

- **Naming Convention**: Matches your project's existing style
  - kebab-case, snake_case, PascalCase, camelCase

## Installation

### From PyPI (Recommended)

```bash
# Using uv (recommended)
uv pip install nanobanana-pro-mcp

# Or using pip
pip install nanobanana-pro-mcp
```

### Prerequisites

- Python 3.10+
- [uv](https://github.com/astral-sh/uv) (recommended) or pip

### Configure Claude Code

Add to your Claude Code MCP settings (`~/.claude.json` or project's `.mcp.json`):

```json
{
  "mcpServers": {
    "nanobanana": {
      "command": "uvx",
      "args": [
        "nanobanana-pro-mcp",
        "--api-key=YOUR_API_KEY",
        "--base-url=http://127.0.0.1:8045/v1"
      ]
    }
  }
}
```

### Configuration Options

| Argument | Required | Description |
|----------|----------|-------------|
| `--api-key` | Yes | Your API key for the image generation service |
| `--base-url` | Yes | Base URL of the OpenAI-compatible image API |
| `--model` | No | Model name (default: `gemini-3-pro-image`) |

## Usage Examples

### Generate an Image

```
"Generate a futuristic city skyline at sunset"
```

The MCP will:
1. Detect this should be a wide/banner image (16:9)
2. Generate the image via your configured API
3. Return a temporary file path and suggested filename

### Convert Format

```
"Convert /tmp/futuristic-city.png to WebP"
```

For icons:
```
"Convert /tmp/app-logo.png to ICO"
```

This creates a proper Windows icon with all required sizes (16x16 through 256x256).

### Save to Project

```
"Save /tmp/futuristic-city.png to my Next.js project at /home/user/my-app"
```

The MCP will:
1. Detect it's a Next.js project
2. Find or create `public/images/`
3. Apply kebab-case naming (or match existing convention)
4. Save as `futuristic-city.png`

## Supported Formats

| Format | Read | Write | Notes |
|--------|------|-------|-------|
| PNG | ✅ | ✅ | Default output format |
| JPG/JPEG | ✅ | ✅ | Converts RGBA to RGB |
| WebP | ✅ | ✅ | Great for web projects |
| ICO | ✅ | ✅ | Multi-size Windows icons |
| ICNS | ✅ | ✅ | macOS icons (basic support) |

## Development

### Local Setup

```bash
git clone https://github.com/YOUR_USERNAME/nanobanana-pro-mcp.git
cd nanobanana-pro-mcp

# Create virtual environment
uv venv
source .venv/bin/activate

# Install in development mode
uv pip install -e .
```

### Testing Locally

```bash
nanobanana-pro-mcp --api-key=test --base-url=http://localhost:8045/v1
```

## License

MIT License - see [LICENSE](LICENSE) for details.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request.
