import json

from src.loader.ImageAnalyser import ImageAnalyser


class UserProfileLoader:
    def __init__(self,model):
        self.model = model
        self.prompt="""
        Analyze this person's photo for fashion styling purposes.

        Return ONLY a JSON object:
        {
            "skin_tone": "fair/light/medium/tan/deep",
            "skin_undertone": "warm/cool/neutral",
            "body_type": "pear/hourglass/rectangle/apple/inverted triangle",
            "face_shape": "oval/round/square/heart/oblong",
            "hair_color": "black/brown/blonde/red/grey/other",
            "best_colors": ["colors that complement skin tone"],
            "avoid_colors": ["colors that clash with skin tone"],
            "best_necklines": ["v-neck/round/sweetheart/etc"],
            "best_silhouettes": ["A-line/flared/straight/etc"],
            "avoid_silhouettes": ["bodycon/boxy/etc"],
            "height_estimate": "petite/medium/tall"
        }

        Return ONLY the JSON. No explanation. No markdown. No backticks.
        """
        self.imageAnalyser = ImageAnalyser(self.model)

    def analyze(self,image_path:str) -> dict:
        document = self.imageAnalyser.analyse_image(image_path, self.prompt)
        return json.loads(document.text)






