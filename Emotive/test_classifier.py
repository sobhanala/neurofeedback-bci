from hybrid_classifier import HybridClassifier

clf = HybridClassifier()

clf.initialize()

result = clf.run_trial()

print(result)

clf.shutdown()