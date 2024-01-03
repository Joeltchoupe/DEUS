def share_data_with_third_party(user_private_key, third_party_did, data_keys):
    # Demander à l'utilisateur s'il est prêt à envoyer ces données
    user_approval = input("Êtes-vous prêt à partager ces données ? (Oui/Non): ").lower()

    # Vérifier l'autorisation de l'utilisateur
    if user_approval != 'oui':
        print("L'autorisation de partage de données a été refusée.")
        return

    # Signature des données avec la clé privée de l'utilisateur
    signature = rsa.sign(str({key: identity_data[key] for key in data_keys}).encode(), user_private_key, 'SHA-1')

    # Envoi des données et de la signature à la tierce partie
    third_party_receive_data(data_keys, identity_data, signature)

# Utilisation de la fonction
share_data_with_third_party(user_private_key, user_did, ['name', 'age'])
