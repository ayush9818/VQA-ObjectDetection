

class ADModule:
    def __init__(self, model_path):
        pass

    def predict(self, image_path, text):
        pass


class VQAModule:
    def __init__(self, model_path):
        pass 

    def predict(self, image_path, text):
        pass



## USAGE 
ad = ADModule(model_path, **kwargs)
vqa = VQAModule(model_path, **kwargs)
answerability = ad.predict(image_path, text)

if answerability:
    ans = vqa.predict(image_path, text)