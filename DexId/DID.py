from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.primitives import serialization

class DexId:
    def __init__(self):
        self.private_key, self.public_key = self.create_did()

    def create_did(self):
        # Génération de la paire de clés elliptiques
        private_key = ec.generate_private_key(ec.SECP256R1())
        public_key = private_key.public_key()

        # Sérialisation des clés au format PEM
        private_key_pem = private_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption()
        )
        public_key_pem = public_key.public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo
        )

        return private_key_pem, public_key_pem

    def register_did_on_blockchain(self):
        # Simulation de l'enregistrement sur la blockchain
        # En pratique, cela nécessiterait une interaction avec la blockchain réelle
        blockchain_transaction = f"Enregistrement DID sur la blockchain : {self.public_key}"

        # Vous devrez remplacer cela par la véritable interaction avec la blockchain

        return blockchain_transaction

# Exemple d'utilisation
dex_id_instance = DexId()
blockchain_transaction_result = dex_id_instance.register_did_on_blockchain()
print(blockchain_transaction_result)

