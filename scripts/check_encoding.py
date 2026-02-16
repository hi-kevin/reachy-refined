import os

def check_file(filepath):
    try:
        with open(filepath, 'rb') as f:
            content = f.read()
            if b'\x00' in content:
                print(f"FAIL: {filepath} contains null bytes!")
                return True
            else:
                # print(f"OK: {filepath}")
                return False
    except Exception as e:
        print(f"ERROR reading {filepath}: {e}")
        return False

print("Checking for null bytes in .py files...")
found = False
for root, dirs, files in os.walk("."):
    for file in files:
        if file.endswith(".py"):
            if check_file(os.path.join(root, file)):
                found = True

if not found:
    print("All .py files look clean (no null bytes).")
