from llama_index.core import Document

from src.loader.ImageAnalyser import ImageAnalyser


class WardrobeAnalyser:
    def __init__(self,model):
        self.model = model
        self.prompt = """
                        Analyze this clothing item image carefully.
                
                        Return ONLY a JSON object with these exact fields:
                        {
                            "type": "kurta/saree/jeans/dress/etc",
                            "color": "primary color",
                            "secondary_colors": ["list", "of", "other", "colors"],
                            "fabric": "silk/cotton/polyester/etc",
                            "style": "anarkali/straight/flared/etc",
                            "occasion": ["festive", "casual", "office", "wedding", "party"],
                            "season": ["summer", "winter", "all"],
                            "pattern": "solid/printed/embroidered/striped/etc",
                            "fit": "loose/regular/fitted",
                            "notes": "any other important styling detail"
                        }
                
                        Return ONLY the JSON. No explanation. No markdown. No backticks.
                        """
        self.imageAnalyser = ImageAnalyser(self.model)

    def analyze(self,image_path:str) -> Document:
        return self.imageAnalyser.analyse_image(image_path,self.prompt)
