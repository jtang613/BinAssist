# BinAssist on Windows - Installation Guide

This guide covers the steps required to get BinAssist and BinAssistMCP plugins working on Windows with Binary Ninja.

## Prerequisites

- Binary Ninja installed (typically at `C:\Program Files\Vector35\BinaryNinja\`)
- Plugins cloned to `%APPDATA%\Binary Ninja\plugins\`
  - `BinAssist\`
  - `BinAssistMCP\`

## The Problem

Binary Ninja on Windows uses a bundled Python environment, but:
1. It doesn't fully process `.pth` files during site initialization
2. The `mcp` library requires `pywin32`, which needs special DLL handling on Windows
3. Packages must be installed to Binary Ninja's user site-packages, not the system Python

## Step 1: Configure Binary Ninja Python Environment

In Binary Ninja, go to **Settings → Python** and configure:
- **Python Environment**: `C:\Program Files\Vector35\BinaryNinja\plugins\python\python310.dll`
- **Site-packages**: `C:/Users/<username>/AppData/Roaming/Binary Ninja/python310/site-packages`

This ensures packages are installed to a user-writable location.

## Step 2: Install pywin32 DLLs (Administrator Required)

The `pywin32` package includes DLLs that must be accessible at runtime. Copy them to the Binary Ninja installation folder.

Open **Command Prompt as Administrator** and run:

```cmd
copy "C:\Users\<username>\AppData\Roaming\Binary Ninja\python310\site-packages\pywin32_system32\*.dll" "C:\Program Files\Vector35\BinaryNinja\"
```

Or if pywin32 is in the Program Files site-packages:

```cmd
copy "C:\Program Files\Vector35\BinaryNinja\plugins\python\Lib\site-packages\pywin32_system32\*.dll" "C:\Program Files\Vector35\BinaryNinja\"
```

This copies `pythoncom310.dll` and `pywintypes310.dll` to a location on the DLL search path.

## Step 3: Install Plugin Dependencies

Install all required packages to Binary Ninja's user site-packages using `--target`:

**Install pywin32:**
```cmd
"C:\Program Files\Vector35\BinaryNinja\plugins\python\python.exe" -m pip install --target "C:\Users\<username>\AppData\Roaming\Binary Ninja\python310\site-packages" pywin32
```

**Install BinAssist dependencies:**
```cmd
"C:\Program Files\Vector35\BinaryNinja\plugins\python\python.exe" -m pip install --target "C:\Users\<username>\AppData\Roaming\Binary Ninja\python310\site-packages" -r "C:\Users\<username>\AppData\Roaming\Binary Ninja\plugins\BinAssist\requirements.txt"
```

**Install BinAssistMCP dependencies:**
```cmd
"C:\Program Files\Vector35\BinaryNinja\plugins\python\python.exe" -m pip install --target "C:\Users\<username>\AppData\Roaming\Binary Ninja\python310\site-packages" -r "C:\Users\<username>\AppData\Roaming\Binary Ninja\plugins\BinAssistMCP\requirements.txt"
```

> **Note:** Replace `<username>` with your actual Windows username, or use `%APPDATA%` in the path (though `--target` may not expand environment variables).

## Step 4: Restart Binary Ninja

Close all Binary Ninja instances and restart. The plugins should now load correctly.

## Verification

In Binary Ninja's Python console, verify the imports work:

```python
import pywintypes
print("pywintypes OK")

import mcp
print("mcp OK")
```

## Notes: pywin32 Path Fix (Already Applied - For Reference Only)

Binary Ninja doesn't process `.pth` files, so the `win32` and `win32\lib` directories aren't automatically added to `sys.path`. 

Both plugins include a fix in their `__init__.py` that adds these paths before importing the `mcp` library:

```python
# Fix pywin32 paths on Windows (required for mcp library)
# Binary Ninja doesn't process .pth files, so we add the paths manually
import sys
import os
if sys.platform == 'win32':
    _site_packages = os.path.join(os.environ.get('APPDATA', ''), 'Binary Ninja', 'python310', 'site-packages')
    _win32_paths = [
        os.path.join(_site_packages, 'win32'),
        os.path.join(_site_packages, 'win32', 'lib'),
        os.path.join(_site_packages, 'Pythonwin'),
    ]
    for _p in _win32_paths:
        if _p not in sys.path and os.path.isdir(_p):
            sys.path.insert(0, _p)
```

This code is already included in the plugin source.


## Troubleshooting

### "ModuleNotFoundError: No module named 'pywintypes'"

- Ensure the DLLs were copied to `C:\Program Files\Vector35\BinaryNinja\`
- Verify the path fix is in `__init__.py` and runs before any mcp imports
- Check that `win32\lib\pywintypes.py` exists in your site-packages

### "Access is denied" when installing packages

- Use the `--target` flag to install to the roaming site-packages
- Or run the command prompt as Administrator

### Packages installing to wrong location

- Always use the full path to Binary Ninja's Python:
  ```
  "C:\Program Files\Vector35\BinaryNinja\plugins\python\python.exe"
  ```
- Use `--target` to explicitly specify the destination

### Check sys.path

In Binary Ninja's Python console:

```python
import sys
print('\n'.join(sys.path))
```

Ensure your roaming site-packages and the win32 subdirectories are listed.
