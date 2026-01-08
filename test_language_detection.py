import unittest

import main


class LanguageDetectionTests(unittest.TestCase):
    def test_roman_urdu_is_flagged(self):
        # Typical roman-urdu phrasing that Twilio STT can produce
        self.assertTrue(main.is_likely_non_english("mujhe madad chahiye"))
        self.assertTrue(main.is_likely_non_english("assalamualaikum"))
        self.assertTrue(main.is_likely_non_english("haan theek hai"))

    def test_other_languages_are_flagged(self):
        self.assertTrue(main.is_likely_non_english("hola necesito ayuda con mi pedido"))
        self.assertTrue(main.is_likely_non_english("bonjour je voudrais de l'aide"))
        self.assertTrue(main.is_likely_non_english("ich brauche hilfe mit meiner bestellung"))

    def test_english_not_flagged(self):
        self.assertFalse(main.is_likely_non_english("Hello, I need help with my order"))
        self.assertFalse(main.is_likely_non_english("My email is test@example.com"))
        self.assertFalse(main.is_likely_non_english("12345"))


if __name__ == "__main__":
    unittest.main()
