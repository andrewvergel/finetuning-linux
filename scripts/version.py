#!/usr/bin/env python3
"""
Version Management Script for Fine-tuning Scripts
Version: 1.0.0
Author: Auto-generated from INSTRUCTIONS.md

This script helps manage version tracking for all fine-tuning scripts.
"""

import os
import json
import re
from datetime import datetime
from pathlib import Path

# Version information
SCRIPT_VERSION = "1.0.0"
SCRIPT_NAME = "version.py"

# Version tracking file
VERSION_FILE = "scripts/versions.json"

# List of scripts to track
SCRIPTS = [
    "finetune_lora.py",
    "inference_lora.py", 
    "merge_adapter.py"
]

def load_version_history():
    """Load version history from file"""
    if os.path.exists(VERSION_FILE):
        with open(VERSION_FILE, 'r') as f:
            return json.load(f)
    return {"scripts": {}, "history": []}

def save_version_history(history):
    """Save version history to file"""
    with open(VERSION_FILE, 'w') as f:
        json.dump(history, f, indent=2)

def get_current_version(script_path):
    """Extract current version from script file"""
    try:
        with open(script_path, 'r') as f:
            content = f.read()
        
        # Look for version pattern in docstring or comments
        version_patterns = [
            r'Version:\s*([\d.]+)',
            r'# Version:\s*([\d.]+)',
            r'VERSION\s*=\s*["\']([^"\']+)["\']'
        ]
        
        for pattern in version_patterns:
            match = re.search(pattern, content)
            if match:
                return match.group(1)
        
        return "0.0.0"  # Default if no version found
    except Exception as e:
        print(f"⚠️  Error reading {script_path}: {e}")
        return "0.0.0"

def update_script_version(script_path, new_version):
    """Update version in script file"""
    try:
        with open(script_path, 'r') as f:
            content = f.read()
        
        # Update version in docstring
        docstring_pattern = r'(Version:\s*)([\d.]+)'
        content = re.sub(docstring_pattern, f'\\g<1>{new_version}', content)
        
        # Update version in comments
        comment_pattern = r'(# Version:\s*)([\d.]+)'
        content = re.sub(comment_pattern, f'\\g<1>{new_version}', content)
        
        # Update VERSION constant if exists
        version_const_pattern = r'(VERSION\s*=\s*["\'])([^"\']+)(["\'])'
        content = re.sub(version_const_pattern, f'\\g<1>{new_version}\\g<3>', content)
        
        with open(script_path, 'w') as f:
            f.write(content)
        
        return True
    except Exception as e:
        print(f"❌ Error updating {script_path}: {e}")
        return False

def show_version_info():
    """Show current version information for all scripts"""
    print(f"📋 {SCRIPT_NAME} v{SCRIPT_VERSION}")
    print(f"📅 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("\n" + "="*60)
    print("🔍 SCRIPT VERSION STATUS")
    print("="*60)
    
    history = load_version_history()
    
    for script in SCRIPTS:
        script_path = os.path.join("scripts", script)
        if os.path.exists(script_path):
            current_version = get_current_version(script_path)
            print(f"📄 {script:<20} v{current_version}")
        else:
            print(f"❌ {script:<20} NOT FOUND")
    
    # Show version history
    if history.get("history"):
        print(f"\n📚 VERSION HISTORY:")
        for entry in history["history"][-5:]:  # Show last 5 entries
            print(f"   {entry['date']}: {entry['script']} v{entry['version']} - {entry['change']}")

def bump_version(script_name, change_type="patch"):
    """Bump version of a specific script"""
    script_path = os.path.join("scripts", script_name)
    
    if not os.path.exists(script_path):
        print(f"❌ Script {script_name} not found")
        return False
    
    current_version = get_current_version(script_path)
    
    # Parse version
    parts = current_version.split('.')
    if len(parts) != 3:
        print(f"❌ Invalid version format: {current_version}")
        return False
    
    major, minor, patch = map(int, parts)
    
    # Bump version
    if change_type == "major":
        major += 1
        minor = 0
        patch = 0
    elif change_type == "minor":
        minor += 1
        patch = 0
    else:  # patch
        patch += 1
    
    new_version = f"{major}.{minor}.{patch}"
    
    # Update script
    if update_script_version(script_path, new_version):
        print(f"✅ Updated {script_name}: v{current_version} → v{new_version}")
        
        # Update history
        history = load_version_history()
        history["scripts"][script_name] = new_version
        history["history"].append({
            "date": datetime.now().isoformat(),
            "script": script_name,
            "version": new_version,
            "change": change_type
        })
        save_version_history(history)
        
        return True
    else:
        print(f"❌ Failed to update {script_name}")
        return False

def show_help():
    """Show help information"""
    print(f"""
🔧 {SCRIPT_NAME} v{SCRIPT_VERSION} - Version Management Tool

USAGE:
    python scripts/version.py [command] [script] [type]

COMMANDS:
    status          Show current version status of all scripts
    bump           Bump version of a specific script
    history        Show version history
    help           Show this help message

EXAMPLES:
    python scripts/version.py status
    python scripts/version.py bump finetune_lora.py patch
    python scripts/version.py bump inference_lora.py minor
    python scripts/version.py bump merge_adapter.py major

VERSION TYPES:
    patch          Increment patch version (1.0.0 → 1.0.1)
    minor          Increment minor version (1.0.0 → 1.1.0)
    major          Increment major version (1.0.0 → 2.0.0)
""")

def main():
    """Main version management function"""
    import sys
    
    if len(sys.argv) < 2:
        show_help()
        return
    
    command = sys.argv[1].lower()
    
    if command == "status":
        show_version_info()
    
    elif command == "bump":
        if len(sys.argv) < 3:
            print("❌ Please specify script name")
            print("Example: python scripts/version.py bump finetune_lora.py patch")
            return
        
        script_name = sys.argv[2]
        change_type = sys.argv[3] if len(sys.argv) > 3 else "patch"
        
        if change_type not in ["patch", "minor", "major"]:
            print("❌ Invalid change type. Use: patch, minor, or major")
            return
        
        bump_version(script_name, change_type)
    
    elif command == "history":
        history = load_version_history()
        if history.get("history"):
            print("📚 VERSION HISTORY:")
            for entry in history["history"]:
                print(f"   {entry['date'][:10]}: {entry['script']} v{entry['version']} ({entry['change']})")
        else:
            print("📚 No version history found")
    
    elif command == "help":
        show_help()
    
    else:
        print(f"❌ Unknown command: {command}")
        show_help()

if __name__ == "__main__":
    main()
