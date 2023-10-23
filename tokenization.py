import re

class tokenizer:
    def substitute(self,text):
        txt = text.lower()
        txt = re.sub('(https?:\/\/|www\.)?\w+[a-zA-Z0-9]{1,}\.[a-zA-Z0-9]{1,}\S+','<URL>',txt) #URL
        txt = re.sub('#\w+','<HASHTAG>,',txt) #HASHTAG
        txt = re.sub('@\w+','<MENTION>',txt) #MENTION
        txt = re.sub(r'\d+', '<NUMBER>', txt)
        return txt

    def Punctuations(self,text):
        # txt = re.sub('\.\B',r' ',text) # examples: Square._
                # replace any repeated punctuation to a single one
        txt = re.sub(r'([.,!?:;—><])\1+', r'\1', text)

        # replace any letter repeated more than twice to a single one (eg: tiired to tired. oopppps -> oops)
        txt = re.sub(r'([a-zA-Z])\1{2,}', r'\1', txt)

        # replace can't with can not
        txt = re.sub(r'can\'t', 'can not', txt)

        # replace xn't with x + not
        txt = re.sub(r'n\'t', r' not', txt)

        # replace x'm with x + am
        txt = re.sub(r'\'m', r' am', txt)

        # replace x's with x + is
        txt = re.sub(r'\'s', r' is', txt)

        txt = re.sub(r'\’s',r' is',txt)

        # replace x're with x + are
        txt = re.sub(r'\'re', r' are', txt)

        # replace x'll to x + will
        txt = re.sub(r'\'ll', r' will', txt)

        # replace x'd to x + would
        txt = re.sub(r'\'d', r' would', txt)

        # replace x've to x + have
        txt = re.sub(r'\'ve', r' have', txt)
        txt = re.sub(r'(\w+)([.,!?:;\[\]*/"\'\(\)])', r'\1 \2', txt)
        txt = re.sub(r'([.,!?:;\[\]*/"\'\(\)])(\w+)', r'\1 \2', txt)
        # remove extra spaces
        txt = re.sub(r'\s+', ' ', txt)
        # split words with hyphens
        txt = re.sub(r'(\w+)?-(\w+)?', r'\1 \2', txt)
        txt = re.sub('[\!\"\$\%\&\'\[\]\(\)\{\}\*\+\,\-\—\/:;=\?\^\_\~\n]',r' ',txt)
        txt = re.sub(r'([\.])', r' \1 ', txt)
        txt = re.sub(r'\s+',' ',txt)
        txt = re.sub(r'\Z','',txt)
        return txt
    def WordTokenizer(self,text):
        return re.split('[\s]',text)
    def random(self,text):
        # s = re.sub(r'([,\.!\?\-;:"&\+\(\)/\[\]])', r' \1 ', text)
        # txt = re.sub(r'([\.])', r' \1 ', text)
        # txt = re.sub('Mr\.','Mr',text)
        # txt = re.sub('Mrs\.','Mrs',txt)
        # txt = re.sub('No\.','No',txt)
        # txt = re.sub('\B\n{1}',' ',txt)
        # txt = re.split('\.\n+|\.\s{1,}|\n{2,}',txt)
        pass
