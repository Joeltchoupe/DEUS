from cryptography.fernet import Fernet
import json
import hashlib

# Fonction pour créer une clé DID (exemple simple, une vraie implémentation serait plus complexe)
def create_did():
    return hashlib.sha256("clé_privée_secrète".encode()).hexdigest()

# Fonction pour crypter les données avec la clé DID
def encrypt_data(data, did):
    key = hashlib.sha256(did.encode()).digest()
    cipher_suite = Fernet(key)
    encrypted_data = cipher_suite.encrypt(json.dumps(data).encode())
    return encrypted_data

# Fonction pour préparer les données et les enregistrer dans un fichier
import time

def generate_filename(account_id):
    timestamp = int(time.time())
    filename = f"data_{account_id}_{timestamp}.txt"
    return filename

# Exemple d'utilisation
#user_account_id = "user123"
#file_name = generate_filename(user_account_id)
#print(file_name)

def prepare_and_save_data(data, did, filename):
    encrypted_data = encrypt_data(data, did)
    with open(filename, 'wb') as file:
        file.write(encrypted_data)

# Exemple d'utilisation
#user_did = create_did()
#user_data = {'name': 'John Doe', 'age': 30, 'address': '123 Main St'}

#prepare_and_save_data(user_data, user_did, 'user_data.txt')

# À ce stade, 'user_data.txt' contient les données cryptées prêtes à être diffusées sur le réseau.
