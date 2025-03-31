from cryptography.fernet import Fernet

# Generate a Fernet key
key = Fernet.generate_key()

# Print the key (it will be a base64-encoded string)
print(key.decode())