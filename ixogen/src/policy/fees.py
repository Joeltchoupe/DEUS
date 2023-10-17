class TxConfirmStats:
    def __init__(self, buckets, bucketMap, decay, scale):
        self.buckets = buckets
        self.bucketMap = bucketMap
        self.decay = decay
        self.scale = scale

        self.txCtAvg = [0.0] * len(buckets)
        self.confAvg = [[0.0] * len(buckets) for _ in range(scale)]
        self.failAvg = [[0.0] * len(buckets) for _ in range(scale)]
        self.m_feerate_avg = [0.0] * len(buckets)
        self.unconfTxs = [[0 for _ in range(len(buckets))] for _ in range(scale)]
        self.oldUnconfTxs = [0 for _ in range(len(buckets))]

    def resizeInMemoryCounters(self, newbuckets):
        self.txCtAvg = [0.0] * newbuckets
        self.confAvg = [[0.0] * newbuckets for _ in range(self.scale)]
        self.failAvg = [[0.0] * newbuckets for _ in range(self.scale)]
        self.m_feerate_avg = [0.0] * newbuckets
        self.unconfTxs = [[0 for _ in range(newbuckets)] for _ in range(self.scale)]
        self.oldUnconfTxs = [0 for _ in range(newbuckets)]

#----------------------------------------------------------------------------------------------

#class TxConfirmStats:

    def EstimateMedianVal(self, confTarget, sufficientTxVal, successBreakPoint, nBlockHeight, result=None):
        """
        Returns the feerate estimate for the given confirmation target.

        Args:
            confTarget: The target number of confirmations.
            sufficientTxVal: The required average number of transactions per block in a bucket range.
            successBreakPoint: The success probability required.
            nBlockHeight: The current block height.
            result: An optional EstimationResult object to populate with the results of the estimation.

        Returns:
            The feerate estimate, or -1 on error conditions.
        """

        # Counters for a bucket (or range of buckets)
        nConf = 0.0  # Number of tx's confirmed within the confTarget
        totalNum = 0.0  # Total number of tx's that were ever confirmed
        extraNum = 0  # Number of tx's still in mempool for confTarget or longer
        failNum = 0.0  # Number of tx's that were never confirmed but removed from the mempool after confTarget
        periodTarget = (confTarget + self.scale - 1) // self.scale
        maxbucketindex = len(self.buckets) - 1

        # We'll combine buckets until we have enough samples.
        # The near and far variables will define the range we've combined
        # The best variables are the last range we saw which still had a high
        # enough confirmation rate to count as success.
        # The cur variables are the current range we're counting.
        curNearBucket = maxbucketindex
        bestNearBucket = maxbucketindex
        curFarBucket = maxbucketindex
        bestFarBucket = maxbucketindex

        foundAnswer = False
        bins = len(self.unconfTxs)
        newBucketRange = True
        passing = True
        passBucket = EstimatorBucket()
        failBucket = EstimatorBucket()

        # Start counting from highest feerate transactions
        for bucket in range(maxbucketindex, -1, -1):
            if newBucketRange:
                curNearBucket = bucket
                newBucketRange = False
            curFarBucket = bucket
            nConf += self.confAvg[periodTarget - 1][bucket]
            totalNum += self.txCtAvg[bucket]
            failNum += self.failAvg[periodTarget - 1][bucket]
            for confct in range(confTarget, self.GetMaxConfirms()):
                extraNum += self.unconfTxs[(nBlockHeight - confct) % bins][bucket]
            extraNum += self.oldUnconfTxs[bucket]
            # If we have enough transaction data points in this range of buckets,
            # we can test for success
            # (Only count the confirmed data points, so that each confirmation count
            # will be looking at the same amount of data and same bucket breaks)
            if totalNum >= sufficientTxVal / (1 - self.decay):
                curPct = nConf / (totalNum + failNum + extraNum)

                # Check to see if we are no longer getting confirmed at the success rate
                if curPct < successBreakPoint:
                    if passing:
                        # First time we hit a failure record the failed bucket
                        failMinBucket = min(curNearBucket, curFarBucket)
                        failMaxBucket = max(curNearBucket, curFarBucket)
                        failBucket.start = failMinBucket if failMinBucket else 0
                        failBucket.end = self.buckets[failMaxBucket]
                        failBucket.withinTarget = nConf
                        failBucket.totalConfirmed = totalNum
                        failBucket.inMempool = extraNum
                        failBucket.leftMempool = failNum
                        passing = False
                    continue

                # Otherwise update the cumulative stats, and the bucket variables
                # and reset the counters
                else:
                    failBucket = EstimatorBucket()  # Reset any failed bucket, currently passing
                    foundAnswer = True
                    passing = True
                    passBucket.withinTarget = nConf
                    nConf = 0
                    passBucket.totalConfirmed = totalNum
                    totalNum = 0
                    passBucket.inMempool = extraNum
                    passBucket.leftMempool = failNum
                    failNum = 0
                    extraNum = 0
                    bestNearBucket = curNearBucket
                    bestFarBucket = curFarBucket

              # Calculate the "average" feerate of the best bucket range that met success conditions
        # Find the bucket with the median transaction and then report the average feerate from that bucket
        # This is a compromise between finding the median which we can't since we don't save all tx's
        # and reporting the average which is less accurate
        unsigned int minBucket = min(bestNearBucket, bestFarBucket);
        unsigned int maxBucket = max(bestNearBucket, bestFarBucket);
        for unsigned int j = minBucket; j <= maxBucket; j++) {
            txSum += self.txCtAvg[j];
        }
        if foundAnswer and txSum != 0:
            txSum = txSum / 2;
            for unsigned int j = minBucket; j <= maxBucket; j++) {
                if self.txCtAvg[j] < txSum:
                    txSum -= self.txCtAvg[j];
                else:  # we're in the right bucket
                    median = self.m_feerate_avg[j] / self.txCtAvg[j];
                    break;
            }

            passBucket.start = minBucket ? self.buckets[minBucket-1] : 0;
            passBucket.end = self.buckets[maxBucket];
        
        # If we were passing until we reached last few buckets with insufficient data, then report those as failed
        if passing and !newBucketRange:
            unsigned int failMinBucket = min(curNearBucket, curFarBucket);
            unsigned int failMaxBucket = max(curNearBucket, curFarBucket);
            failBucket.start = failMinBucket ? self.buckets[failMinBucket-1] : 0;
            failBucket.end = self.buckets[failMaxBucket];
            failBucket.withinTarget = nConf;
            failBucket.totalConfirmed = totalNum;
            failBucket.inMempool = extraNum;
            failBucket.leftMempool = failNum;
        
        float passed_within_target_perc = 0.0;
        float failed_within_target_perc = 0.0;
        if ((passBucket.totalConfirmed + passBucket.inMempool + passBucket.leftMempool)):
            passed_within_target_perc = 100 * passBucket.withinTarget / (passBucket.totalConfirmed + passBucket.inMempool + passBucket.leftMempool);
        
        if ((failBucket.totalConfirmed + failBucket.inMempool + failBucket.leftMempool)):
            failed_within_target_perc = 100 * failBucket.withinTarget / (failBucket.totalConfirmed + failBucket.inMempool + failBucket.leftMempool);

        LogPrint(BCLog::ESTIMATEFEE, "FeeEst: %d > %.0f%% decay %.5f: feerate: %g from (%g - %g) %.2f%% %.1f/(%.1f %d mem %.1f out) Fail: (%g - %g) %.2f%% %.1f/(%.1f %d mem %.1f out)\n",
             confTarget, 100.0 * successBreakPoint, decay,
             median, passBucket.start, passBucket.end,
             passed_within_target_perc,
             passBucket.withinTarget, passBucket.totalConfirmed, passBucket.inMempool, passBucket.leftMempool,
             failBucket.start, failBucket.end,
             failed_within_target_perc,
             failBucket.withinTarget, failBucket.totalConfirmed, failBucket.inMempool, failBucket.leftMempool);


        if result:
            result->pass = passBucket;
            result->fail = failBucket;
            result->decay = decay;
            result->scale = scale;
        return median;

#--------------------------------------------------------------------------------------------------------------------------------------------

import fs
import math

class CBlockPolicyEstimator:

    def __init__(self, estimation_filepath, read_stale_estimates=False):
        self.m_estimation_filepath = estimation_filepath
        self.nBestSeenHeight = 0
        self.untrackedTxs = 0
        self.trackedTxs = 0

        self.buckets = []
        self.bucketMap = {}
        for bucketBoundary in range(1, math.ceil(MAX_BUCKET_FEERATE / FEE_SPACING) + 1):
            self.buckets.append(bucketBoundary * FEE_SPACING)
            self.bucketMap[bucketBoundary * FEE_SPACING] = bucketBoundary
        self.buckets.append(math.inf)
        self.bucketMap[math.inf] = len(self.buckets) - 1
        assert len(self.bucketMap) == len(self.buckets)

        self.feeStats = TxConfirmStats(self.buckets, self.bucketMap, MED_BLOCK_PERIODS, MED_DECAY, MED_SCALE)
        self.shortStats = TxConfirmStats(self.buckets, self.bucketMap, SHORT_BLOCK_PERIODS, SHORT_DECAY, SHORT_SCALE)
        self.longStats = TxConfirmStats(self.buckets, self.bucketMap, LONG_BLOCK_PERIODS, LONG_DECAY, LONG_SCALE)

        try:
            est_file = fs.fopen(self.m_estimation_filepath, "rb")
        except FileNotFoundError:
            LogPrintf("%s is not found. Continue anyway.\n", self.m_estimation_filepath)
            return

        file_age = (datetime.datetime.now() - est_file.stat().st_ctime).total_seconds() / 3600
        if file_age > MAX_FILE_AGE and not read_stale_estimates:
            LogPrintf("Fee estimation file %s too old (age=%lld > %lld hours) and will not be used to avoid serving stale estimates.\n", self.m_estimation_filepath, file_age, MAX_FILE_AGE)
            return

        if not self.Read(est_file):
            LogPrintf("Failed to read fee estimates from %s. Continue anyway.\n", self.m_estimation_filepath)

    def processTransaction(self, entry, validFeeEstimate):
        with lock(self):
            txHeight = entry.GetHeight()
            hash = entry.GetTx().GetHash()
            if hash in self.mapMemPoolTxs:
                LogPrint(BCLog::ESTIMATEFEE, "Blockpolicy error mempool tx %s already being tracked\n", hash)
                return

            if txHeight != self.nBestSeenHeight:
                # Ignore side chains and re-orgs; assuming they are random they don't
                # affect the estimate.  We'll potentially double count transactions in 1-block reorgs.
                # Ignore txs if BlockPolicyEstimator is not in sync with ActiveChain().Tip().
                # It will be synced next time a block is processed.
                return

            # Only want to be updating estimates when our blockchain is synced,
            # otherwise we'll miscalculate how many blocks its taking to get included.
            if not validFeeEstimate:
                self.untrackedTxs += 1
                return

            self.trackedTxs += 1

    def processBlockTx(self, nBlockHeight, entry):
        with lock(self):
            # ...

            # Feerates are stored and reported as BTC-per-kb:
            feeRate = CFeeRate(entry.GetFee(), entry.GetTxSize())

            self.feeStats.Record(blocksToConfirm, feeRate.GetFeePerK())
            self.shortStats.Record(blocksToConfirm, feeRate.GetFeePerK())
            self.longStats.Record(blocksToConfirm, feeRate.GetFeePerK())

    def processBlock(self, nBlockHeight, entries):
        with lock(self):
            if nBlockHeight <= self.nBestSeenHeight:
                # ...
                return

            # Must update nBestSeenHeight in sync with ClearCurrent so that
            # calls to removeTx (via processBlockTx) correctly calculate age
            # of unconfirmed txs to remove from tracking.
            self.nBestSeenHeight = nBlockHeight

            # Update unconfirmed circular buffer
            self.feeStats.ClearCurrent(nBlockHeight)
            self.shortStats.ClearCurrent(nBlockHeight)
            self.longStats.ClearCurrent(nBlockHeight)

            # Decay all exponential averages
            self.feeStats.UpdateMovingAverages()
            self.shortStats.UpdateMovingAverages()
            self.longStats.UpdateMovingAverages()

            countedTxs = 0
            # Update averages with data points from current block
            for entry in entries:
                if self.processBlockTx(nBlockHeight, entry):
                    countedTxs += 1

            if self.firstRecordedHeight == 0 and countedTxs > 0:
                self.firstRecordedHeight = nBlockHeight
                LogPrint(BCLog::ESTIMATEFEE, "Blockpolicy first recorded height %u\n", self.firstRecordedHeight)

            LogPrint(BCLog::ESTIMATEFEE, "Blockpolicy estimates updated by %u of %u block txs, since last block %u of %u tracked, mempool map size %u, max target %u from %s\n",
                     countedTxs, len(entries), self.trackedTxs, self.trackedTxs + self.untrackedTxs, len(self.mapMemPoolTxs),
                     self.MaxUsableEstimate(), self.HistoricalBlockSpan() > self.BlockSpan() and "historical" or "current")

            self.trackedTxs = 0
            self.untrackedTxs = 0

    def estimateFee(self, confTarget):
        # It's not possible to get reasonable estimates for confTarget of 1
        if confTarget <= 1:
            return CFeeRate(0)

        return self.estimateRawFee(confTarget, DOUBLE_SUCCESS_PCT, FeeEstimateHorizon.MED_HALFLIFE)

    def estimateRawFee(self, confTarget, successThreshold, horizon, result=None):
        stats = None
        sufficientTxs = SUFFICIENT_FEETXS
        if horizon == FeeEstimateHorizon.SHORT_HALFLIFE:
            stats = self.shortStats
            sufficientTxs = SUFFICIENT_TXS_SHORT
        elif horizon == FeeEstimateHorizon.MED_HALFLIFE:
            stats = self.feeStats
        elif horizon == FeeEstimateHorizon.LONG_HALFLIFE:
            stats = self.longStats

        assert stats

        return stats.EstimateMedianVal(confTarget, sufficientTxs, successThreshold, self.nBestSeenHeight, result)

#----------------------------------------------------------------------------------------------------------------------------

def estimateFeeAtLeast(self, confTarget, minFeeRate):
    """
    Estimate the feerate that a transaction with minFeeRate is likely to be confirmed within confTarget blocks.

    Args:
        confTarget: The number of blocks to confirm.
        minFeeRate: The minimum feerate required.

    Returns:
        The estimated feerate that a transaction with minFeeRate is likely to be confirmed within confTarget blocks.
    """

    # It's not possible to get reasonable estimates for confTarget of 1
    if confTarget <= 1:
        return CFeeRate(0)

    # Estimate the feerate that a transaction with minFeeRate is likely to be confirmed within confTarget blocks.
    median = self.estimateRawFee(confTarget, DOUBLE_SUCCESS_PCT, FeeEstimateHorizon.MED_HALFLIFE)

    # If the median feerate is less than minFeeRate, then the transaction is likely to be confirmed within confTarget blocks.
    return median if median >= minFeeRate else minFeeRate

def estimateMedian(self):
    """
    Estimate the median feerate of transactions that confirmed within the most recent block.

    Returns:
        The estimated median feerate of transactions that confirmed within the most recent block.
    """

    return self.shortStats.EstimateMedian(1, SUFFICIENT_TXS_SHORT, 1.0, self.nBestSeenHeight)

def estimateMedianAtLeast(self, minFeeRate):
    """
    Estimate the median feerate that a transaction with minFeeRate is likely to be confirmed in the next block.

    Args:
        minFeeRate: The minimum feerate required.

    Returns:
        The estimated median feerate that a transaction with minFeeRate is likely to be confirmed in the next block.
    """

    return self.estimateMedian() if self.estimateMedian() >= minFeeRate else minFeeRate

def estimateMax(self):
    """
    Estimate the maximum feerate of transactions that confirmed within the most recent block.

    Returns:
        The estimated maximum feerate of transactions that confirmed within the most recent block.
    """

    return self.shortStats.EstimateMax(1, SUFFICIENT_TXS_SHORT, 1.0, self.nBestSeenHeight)

def estimateMaxAtLeast(self, minFeeRate):
    """
    Estimate the maximum feerate that a transaction with minFeeRate is likely to be confirmed in the next block.

    Args:
        minFeeRate: The minimum feerate required.

    Returns:
        The estimated maximum feerate that a transaction with minFeeRate is likely to be confirmed in the next block.
    """

    return self.estimateMax() if self.estimateMax() >= minFeeRate else minFeeRate

def estimateCombinedFee(self, confTarget, successThreshold, checkShorterHorizon=True, result=None):
        """
        Return a fee estimate at the required successThreshold from the shortest
        time horizon which tracks confirmations up to the desired target.  If
        checkShorterHorizon is requested, also allow short time horizon estimates
        for a lower target to reduce the given answer
        """

        estimate = -1
        if 1 <= confTarget <= self.longStats.GetMaxConfirms():
            # Find estimate from shortest time horizon possible
            if confTarget <= self.shortStats.GetMaxConfirms():  # short horizon
                estimate = self.shortStats.EstimateMedianVal(confTarget, SUFFICIENT_TXS_SHORT, successThreshold, self.nBestSeenHeight, result)
            elif confTarget <= self.feeStats.GetMaxConfirms():  # medium horizon
                estimate = self.feeStats.EstimateMedianVal(confTarget, SUFFICIENT_FEETXS, successThreshold, self.nBestSeenHeight, result)
            else:  # long horizon
                estimate = self.longStats.EstimateMedianVal(confTarget, SUFFICIENT_FEETXS, successThreshold, self.nBestSeenHeight, result)
            if checkShorterHorizon:
                EstimationResult tempResult
                # If a lower confTarget from a more recent horizon returns a lower answer use it.
                if confTarget > self.feeStats.GetMaxConfirms():
                    medMax = self.feeStats.EstimateMedianVal(self.feeStats.GetMaxConfirms(), SUFFICIENT_FEETXS, successThreshold, self.nBestSeenHeight, &tempResult)
                    if medMax > 0 and (estimate == -1 or medMax < estimate):
                        estimate = medMax
                        if result:
                            result = tempResult
                if confTarget > self.shortStats.GetMaxConfirms():
                    shortMax = self.shortStats.EstimateMedianVal(self.shortStats.GetMaxConfirms(), SUFFICIENT_TXS_SHORT, successThreshold, self.nBestSeenHeight, &tempResult)
                    if shortMax > 0 and (estimate == -1 or shortMax < estimate):
                        estimate = shortMax
                        if result:
                            result = tempResult
        return estimate
def estimateSmartFee(self, confTarget, feeCalc=None, conservative=False):
        """
        estimateSmartFee returns the max of the feerates calculated with a 60%
        threshold required at target / 2, an 85% threshold required at target and a
        95% threshold required at 2 * target.  Each calculation is performed at the
        shortest time horizon which tracks the required target.  Conservative
        estimates, however, required the 95% threshold at 2 * target be met for any
        longer time horizons also.
        """

        with lock(self):

            if feeCalc:
                feeCalc.desiredTarget = confTarget
                feeCalc.returnedTarget = confTarget

            median = -1
            tempResult = EstimationResult()

            # Return failure if trying to analyze a target we're not tracking
            if confTarget <= 0 or (unsigned int)confTarget > self.longStats.GetMaxConfirms():
                return CFeeRate(0)  # error condition

            # It's not possible to get reasonable estimates for confTarget of 1
            if confTarget == 1:
                confTarget = 2

            maxUsableEstimate = self.MaxUsableEstimate()
            if (unsigned int)confTarget > maxUsableEstimate:
                confTarget = maxUsableEstimate
            if feeCalc:
                feeCalc.returnedTarget = confTarget

            if confTarget <= 1:
                return CFeeRate(0)  # error condition

            assert confTarget > 0  # estimateCombinedFee and estimateConservativeFee take unsigned ints/** true is passed to estimateCombined fee for target/2 and target so that we check the max confirms for shorter time horizons as well.
             #* This is necessary to preserve monotonically increasing estimates.
             #* For non-conservative estimates we do the same thing for 2*target, but
             #"* for conservative estimates we want to skip these shorter horizons
             #* checks for 2*target because we are taking the max over all time
             #* horizons so we already have monotonically increasing estimates and
             #* the purpose of conservative estimates is not to let short term
             #* fluctuations lower our estimates by too much.
             #*/
             #*
            halfEst = self.estimateCombinedFee(confTarget/2, HALF_SUCCESS_PCT, true, &tempResult)
            if feeCalc:
                feeCalc.est = tempResult
                feeCalc.reason = FeeReason.HALF_ESTIMATE
            median = halfEst
            actualEst = self.estimateCombinedFee(confTarget, SUCCESS_PCT, true, &tempResult)
            if actualEst > median:
                median = actualEst
                if feeCalc:
                    feeCalc.est = tempResult
                    feeCalc.reason = FeeReason.FULL_ESTIMATE
            doubleEst = self.estimateCombinedFee(2 * confTarget, DOUBLE_SUCCESS_PCT, !conservative, &tempResult)
            if doubleEst > median:
                median = doubleEst
                if feeCalc:
                    feeCalc.est = tempResult
                    feeCalc.reason = FeeReason.DOUBLE_ESTIMATE

            if conservative or median == -1:
                consEst = self.estimateConservativeFee(2 * confTarget, &tempResult)
                if consEst > median:
                    median = consEst
                    if feeCalc:
                        feeCalc.est = tempResult
                        feeCalc.reason = FeeReason.CONSERVATIVE

            if median < 0:
                return CFeeRate(0)  # error condition

            return CFeeRate(llround(median))

  def FlushUnconfirmed(self):
        """
        Remove every entry in mapMemPoolTxs
        """

        with lock(self):
            num_entries = len(self.mapMemPoolTxs)
            startclear = datetime.datetime.now()
            while self.mapMemPoolTxs:
                mi = self.mapMemPoolTxs.begin()
                self._removeTx(mi.first, False)  # this calls erase() on mapMemPoolTxs
            endclear = datetime.datetime.now()
            LogPrint(BCLog.ESTIMATEFEE, "Recorded %u unconfirmed txs from mempool in %gs\n", num_entries, (endclear - startclear).total_seconds())

    def GetFeeEstimatorFileAge(self):
        """
        Return the age of the fee estimator file in hours.
        """

        file_time = self.m_estimation_filepath.last_write_time()
        now = datetime.datetime.now()
        return (now - file_time).total_seconds() / 3600

class FeeFilterRounder:

    def __init__(self, minIncrementalFee, rng):
        """
        FeeFilterRounder is a class that rounds fees to the nearest fee filter bucket.

        Args:
            minIncrementalFee: The minimum incremental fee.
            rng: A random number generator.
        """

        self.m_fee_set = MakeFeeSet(minIncrementalFee, MAX_FILTER_FEERATE, FEE_FILTER_SPACING)
        self.insecure_rand = rng

    def round(self, currentMinFee):
        """
        Round the given fee to the nearest fee filter bucket.

        Args:
            currentMinFee: The fee to round.

        Returns:
            The rounded fee.
        """

        with lock(self.insecure_rand_mutex):
            it = self.m_fee_set.lower_bound(currentMinFee)
            if it == self.m_fee_set.end() or (it != self.m_fee_set.begin() and self.insecure_rand.getrandbits(1) != 0):
                it -= 1
            return int(it)

def MakeFeeSet(min_incremental_fee, max_filter_fee_rate, fee_filter_spacing):
    """
    Create a set of fee filter buckets.

    Args:
        min_incremental_fee: The minimum incremental fee.
        max_filter_fee_rate: The maximum fee filter rate.
        fee_filter_spacing: The fee filter spacing.

    Returns:
        A set of fee filter buckets.
    """

    fee_set = set()

    min_fee_limit = max(1, min_incremental_fee.GetFeePerK() // 2)
    fee_set.add(0)
    for bucket_boundary in range(min_fee_limit, max_filter_fee_rate + 1, fee_filter_spacing):
        fee_set.add(bucket_boundary)

    return fee_set
