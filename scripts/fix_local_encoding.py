import os

def check_file(filepath):
    try:
        with open(filepath, 'rb') as f:
            content = f.read()
        
        if b'\x00' in content:
            print(f"Fixing {filepath} (Null bytes found)...")
            try:
                text = content.decode('utf-16')
            except:
                text = content.replace(b'\x00', b'').decode('utf-8', errors='ignore')
            
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(text)
            print(f"Fixed {filepath}.")
        else:
            # print(f"{filepath} is clean.")
            pass
    except Exception as e:
        print(f"Error checking {filepath}: {e}")

print("Checking all .py files...")
for root, dirs, files in os.walk("."):
    for file in files:
        if file.endswith(".py"):
            check_file(os.path.join(root, file))
