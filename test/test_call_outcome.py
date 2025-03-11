import unittest
from call_outcome_classifier import classify_call_outcome

class TestCallOutcomeClassifier(unittest.TestCase):
    def test_issue_resolved(self):
        text_data = {"test1.txt": "Thank you for your help! Everything is resolved."}
        result = classify_call_outcome(text_data)
        self.assertEqual(result["test1.txt"], "Issue Resolved")
    
    def test_follow_up_needed(self):
        text_data = {"test2.txt": "I still need assistance with my issue."}
        result = classify_call_outcome(text_data)
        self.assertEqual(result["test2.txt"], "Follow-up Action Needed")
    
if __name__ == "__main__":
    unittest.main()