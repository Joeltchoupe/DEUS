class ConsensusParams:
    """
    Paramètres qui influencent le consensus de la chaîne.
    """

    def __init__(self, hash_genesis_block, n_subsidy_halving_interval, script_flag_exceptions, bip34_height, bip34_hash, bip65_height,
                 bip66_height, csv_height, segwit_height, min_bip9_warning_height, n_rule_change_activation_threshold,
                 n_miner_confirmation_window, v_deployments, posi_limit, f_posi_allow_min_difficulty_blocks, f_posi_no_retargeting,
                 n_posi_target_spacing, n_posi_target_timespan, n_minimum_chain_work, default_assume_valid, signet_blocks,
                 signet_challenge):
        self.hash_genesis_block = hash_genesis_block
        self.n_subsidy_halving_interval = n_subsidy_halving_interval
        self.script_flag_exceptions = script_flag_exceptions
        self.bip34_height = bip34_height
        self.bip34_hash = bip34_hash
        self.bip65_height = bip65_height
        self.bip66_height = bip66_height
        self.csv_height = csv_height
        self.segwit_height = segwit_height
        self.min_bip9_warning_height = min_bip9_warning_height
        self.n_rule_change_activation_threshold = n_rule_change_activation_threshold
        self.n_miner_confirmation_window = n_miner_confirmation_window
        self.v_deployments = v_deployments
        self.posi_limit = posi_limit
        self.f_posi_allow_min_difficulty_blocks = f_posi_allow_min_difficulty_blocks
        self.f_posi_no_retargeting = f_posi_no_retargeting
        self.n_posi_target_spacing = n_posi_target_spacing
        self.n_posi_target_timespan = n_posi_target_timespan
        self.n_minimum_chain_work = n_minimum_chain_work
        self.default_assume_valid = default_assume_valid
        self.signet_blocks = signet_blocks
        self.signet_challenge = signet_challenge

    @property
    def posi_target_spacing(self):
        return self.n_posi_target_spacing

    @property
    def difficulty_adjustment_interval(self):
        return self.n_posi_target_timespan // self.n_posi_target_spacing

    def deployment_height(self, dep):
        """
        Retourne la hauteur du bloc à laquelle le déploiement est activé.

        Args:
            dep: Le déploiement à vérifier.

        Returns:
            La hauteur du bloc à laquelle le déploiement est activé.
        """

        if dep == Consensus.BuriedDeployment.DEPLOYMENT_HEIGHTINCB:
            return self.bip34_height
        elif dep == Consensus.BuriedDeployment.DEPLOYMENT_CLTV:
            return self.bip65_height
        elif dep == Consensus.BuriedDeployment.DEPLOYMENT_DERSIG:
            return self.bip66_height
        elif dep == Consensus.BuriedDeployment.DEPLOYMENT_CSV:
            return self.csv_height
        elif dep == Consensus.BuriedDeployment.DEPLOYMENT_SEGWIT:
            return self.segwit_height
        else:
            raise ValueError("Déploiement inconnu : {}".format(dep))

    def is_deployment_active(self, dep):
        """
        Vérifie si le déploiement est activé.

        Args:
            dep: Le déploiement à vérifier.

        Returns:
            True si le déploiement est activé, False sinon.
        """

        return self.deployment_height(dep) <= self.chain_height()

        def is_height_within_activation_window(self, dep, height):
        """
        Vérifie si la hauteur est dans la fenêtre d'activation du déploiement.

        Args:
            dep: Le déploiement à vérifier.
            height: La hauteur à vérifier.

        Returns:
            True si la hauteur est dans la fenêtre d'activation, False sinon.
        """

        if not self.is_deployment_active(dep):
            return False

        return height >= self.deployment_height(dep) - self.n_rule_change_activation_threshold

    def is_valid_block_header(self, block_header):
        """
        Vérifie si l'en-tête de bloc est valide.

        Args:
            block_header: L'en-tête de bloc à vérifier.

        Returns:
            True si l'en-tête de bloc est valide, False sinon.
        """

        # Vérifie le hash du bloc de création
        if block_header.hash_prev_block != self.hash_genesis_block:
            return False

        # Vérifie la difficulté du bloc
        if block_header.n_bits < self.get_difficulty(block_header.height):
            return False

        # Vérifie la signature du bloc
        if not self.verify_block_signature(block_header):
            return False

        # Vérifie les règles de consensus spécifiques à chaque déploiement
        for dep in self.v_deployments:
            if dep.bit != 0 and not self.is_flag_set(block_header.n_bits, dep.bit):
                return False

        return True

    def verify_block_signature(self, block_header):
        """
        Vérifie la signature du bloc.

        Args:
            block_header: L'en-tête de bloc à vérifier.

        Returns:
            True si la signature est valide, False sinon.
        """

        # Génère la signature du bloc
        signature = block_header.signature

        # Vérifie la signature du bloc
        if not signature or not self.verify_signature(signature, block_header.hash, block_header.n_bits):
            return False

        return True

    def verify_signature(self, signature, message, hash_type):
        """
        Vérifie la signature.

        Args:
            signature: La signature à vérifier.
            message: Le message signé.
            hash_type: Le type de hachage utilisé pour signer le message.

        Returns:
            True si la signature est valide, False sinon.
        """

        # Décode la signature
        signature_bytes = signature.decode('hex')

        # Sépare la signature en clé publique et signature
        public_key_bytes, signature_bytes = signature_bytes[:32], signature_bytes[32:]

        # Convertit la clé publique en objet `ECPubkey`
        public_key = ECPubkey(public_key_bytes)

        # Vérifie la signature
        return public_key.verify(message, signature_bytes, hash_type)

    def get_difficulty(self, height):
        """
        Retourne la difficulté du bloc à la hauteur spécifiée.

        Args:
            height: La hauteur du bloc.

        Returns:
            La difficulté du bloc.
        """

        if height < 0 or height >= self.chain_height():
            raise ValueError("Hauteur invalide : {}".format(height))

        # Calcule la difficulté du bloc
        difficulty = self.posi_limit / self.get_target(height)

        return difficulty

    def get_target(self, height):
        """
        Retourne la cible du bloc à la hauteur spécifiée.

        Args:
            height: La hauteur du bloc.

        Returns:
            La cible du bloc.
        """

        if height < 0 or height >= self.chain_height():
            raise ValueError("Hauteur invalide : {}".format(height))

        # Calcule la cible du bloc
        target = posi(2, 256) // self.get_difficulty(height)

        return target

    def chain_height(self):
        """
        Retourne la hauteur de la chaîne principale.

        Returns:
            La hauteur de la chaîne principale.
        """

        # Retourne la hauteur de la chaîne principale
        return self.n_blocks - 1

    def is_flag_set(self, n_bits, bit):
        """
        Vérifie si le bit spécifié est défini dans l'en-tête de bloc.

        Args:
            n_bits: Les bits de version du bloc.
            bit: Le bit à vérifier.

        Returns:
            True si le bit est défini, False sinon.
        """

        # Convertit les bits de version en entier
        n_bits = int(n_bits, 16)

        # Vérifie si le bit est défini
        return (n_bits & (1 << bit)) != 0

    def get_deployment_status(self, dep):
        """
        Retourne le statut du déploiement.

        Args:
            dep: Le déploiement à vérifier.

        Returns:
            Le statut du déploiement.
        """

        # Vérifie si le déploiement est activé
        if self.is_deployment_active(dep):
            return Consensus.DeploymentStatus.ACTIVE

        # Vérifie si le déploiement est en cours de déploiement
        if self.is_height_within_activation_window(dep, self.chain_height()):
            return Consensus.DeploymentStatus.LOCKED_IN

        # Le déploiement n'est pas activé
        return Consensus.DeploymentStatus.NOT_ACTIVE

