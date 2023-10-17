# Vérifie la validité d'un en-tête de bloc
params = ConsensusParams()
block_header = BlockHeader()

if params.is_valid_block_header(block_header):
    print("L'en-tête de bloc est valide")
else:
    print("L'en-tête de bloc est invalide")

# Retourne la difficulté du bloc à la hauteur spécifiée
params = ConsensusParams()
height = 1000

difficulty = params.get_difficulty(height)

print("La difficulté du bloc à la hauteur {} est {}".format(height, difficulty))

# Retourne le statut du déploiement SegWit
params = ConsensusParams()

status = params.get_deployment_status(Consensus.DeploymentPos.SEGWIT)

print("Le statut du déploiement SegWit est {}".format(status))
