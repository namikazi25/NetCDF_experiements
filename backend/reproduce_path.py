import os

UPLOAD_DIR = "uploads"
filename = "schouts_1.nc"

# Simulate app.py logic
file_path = os.path.join(UPLOAD_DIR, filename)
print(f"Relative path: {file_path}")

abs_path = os.path.abspath(file_path)
print(f"Absolute path: {abs_path}")

# Check if file exists
if os.path.exists(abs_path):
    print("File exists at absolute path.")
else:
    print("File DOES NOT exist at absolute path.")

# Simulate the error condition
wrong_path = os.path.join(os.getcwd(), filename)
print(f"Wrong path (simulated): {wrong_path}")
if os.path.exists(wrong_path):
    print("File exists at wrong path.")
else:
    print("File DOES NOT exist at wrong path.")
