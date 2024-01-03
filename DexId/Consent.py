from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import ec

def generate_key_pair():
    private_key = ec.generate_private_key(ec.SECP256R1(), default_backend())
    public_key = private_key.public_key()
    return private_key, public_key

def sign_data(private_key, data):
    signature = private_key.sign(data, ec.ECDSA(hashes.SHA256()))
    return signature

def verify_signature(public_key, data, signature):
    try:
        public_key.verify(signature, data, ec.ECDSA(hashes.SHA256()))
        return True
    except Exception:
        return False

# Exemple d'utilisation
user_private_key, user_public_key = generate_key_pair()
third_party_public_key = generate_key_pair()[1]

consent_data = "Consent for sharing information"
user_signature = sign_data(user_private_key, consent_data)

# Simulation de la validation du consentement par un tiers
is_valid_consent = verify_signature(user_public_key, consent_data, user_signature)

print(f"Is the consent valid? {is_valid_consent}")
