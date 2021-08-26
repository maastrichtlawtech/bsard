from statistics import mean


class Evaluator:
    def __init__(self):
        pass

    def compute_all_metrics(self, ground_truths, results, precision_range=[2,5,10], recall_range=[5,10,20,50,100]):
        metrics = list()
        scores = list()
        
        # Compute MAP and MRR.
        metrics.extend(['MAP', 'MRR'])
        scores.append(self.compute_mean_score(self.average_precision, ground_truths, results))
        scores.append(self.compute_mean_score(self.reciprocal_rank, ground_truths, results))

        # Compute Recall@k.
        for k in recall_range:
            metrics.append('R@'+str(k))
            scores.append(self.compute_mean_score(self.recall, ground_truths, results, at=k))

        # Compute Precision@k.
        for k in precision_range:
            metrics.append('P@'+str(k))
            scores.append(self.compute_mean_score(self.precision, ground_truths, results, at=k))

        return dict(zip(metrics, scores))

    def compute_mean_score(self, func, all_ground_truths, all_results, at=None):
        return mean([func(truths, res, at=at) for truths, res in zip(all_ground_truths, all_results)])

    def precision(self, ground_truths, results, at=None):
        at = len(results) if at is None else at
        relevances = [1 if d in ground_truths else 0 for d in results[:at]]
        return sum(relevances)/len(results[:at])

    def recall(self, ground_truths, results, at=None):
        at = len(results) if at is None else at
        relevances = [1 if d in ground_truths else 0 for d in results[:at]]
        return sum(relevances)/len(ground_truths)

    def fscore(self, ground_truths, results):
        p = self.precision(ground_truths, results)
        r = self.recall(ground_truths, results)
        return (2*p*r)/(p+r) if (p != 0.0 or r != 0.0) else 0.0

    def reciprocal_rank(self, ground_truths, results, at=None):
        at = len(results) if at is None else at
        return max([1/(i+1) if d in ground_truths else 0.0 for i, d in enumerate(results[:at])])

    def average_precision(self, ground_truths, results, at=None):
        at = len(results) if at is None else at
        p_at = [self.precision(ground_truths, results, at=i+1) if d in ground_truths else 0 for i, d in enumerate(results[:at])]
        return sum(p_at)/len(ground_truths)
