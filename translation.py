from langchain.text_splitter import RecursiveCharacterTextSplitter
from transformers import pipeline

model_path = "models/models--facebook--nllb-200-distilled-600M/snapshots/f8d333a098d19b4fd9a8b18f94170487ad3f821d"
pipe = pipeline("translation", model=model_path)

language_code_map = {
    'af': 'afr_Latn',    # Afrikaans
    'am': 'amh_Ethi',    # Amharic
    'ar': 'arb_Arab',    # Arabic
    'as': 'asm_Beng',    # Assamese
    'az': 'azj_Latn',    # Azerbaijani
    'be': 'bel_Cyrl',    # Belarusian
    'bg': 'bul_Cyrl',    # Bulgarian
    'bn': 'ben_Beng',    # Bengali
    'bs': 'bos_Latn',    # Bosnian
    'ca': 'cat_Latn',    # Catalan
    'ceb': 'ceb_Latn',   # Cebuano
    'cs': 'ces_Latn',    # Czech
    'cy': 'cym_Latn',    # Welsh
    'da': 'dan_Latn',    # Danish
    'de': 'deu_Latn',    # German
    'el': 'ell_Grek',    # Greek
    'en': 'eng_Latn',    # English
    'es': 'spa_Latn',    # Spanish
    'et': 'est_Latn',    # Estonian
    'fa': 'pes_Arab',    # Persian
    'fi': 'fin_Latn',    # Finnish
    'fr': 'fra_Latn',    # French
    'ga': 'gle_Latn',    # Irish
    'gl': 'glg_Latn',    # Galician
    'gu': 'guj_Gujr',    # Gujarati
    'ha': 'hau_Latn',    # Hausa
    'hi': 'hin_Deva',    # Hindi
    'hr': 'hrv_Latn',    # Croatian
    'hu': 'hun_Latn',    # Hungarian
    'hy': 'hye_Armn',    # Armenian
    'id': 'ind_Latn',    # Indonesian
    'ig': 'ibo_Latn',    # Igbo
    'is': 'isl_Latn',    # Icelandic
    'it': 'ita_Latn',    # Italian
    'ja': 'jpn_Jpan',    # Japanese
    'jv': 'jav_Latn',    # Javanese
    'ka': 'kat_Geor',    # Georgian
    'kk': 'kaz_Cyrl',    # Kazakh
    'km': 'khm_Khmr',    # Khmer
    'kn': 'kan_Knda',    # Kannada
    'ko': 'kor_Hang',    # Korean
    'lb': 'ltz_Latn',    # Luxembourgish
    'ln': 'lin_Latn',    # Lingala
    'lo': 'lao_Laoo',    # Lao
    'lt': 'lit_Latn',    # Lithuanian
    'lv': 'lvs_Latn',    # Latvian
    'mg': 'plt_Latn',    # Malagasy
    'mk': 'mkd_Cyrl',    # Macedonian
    'ml': 'mal_Mlym',    # Malayalam
    'mn': 'khk_Cyrl',    # Mongolian
    'mr': 'mar_Deva',    # Marathi
    'ms': 'zsm_Latn',    # Malay
    'mt': 'mlt_Latn',    # Maltese
    'my': 'mya_Mymr',    # Burmese
    'ne': 'npi_Deva',    # Nepali
    'nl': 'nld_Latn',    # Dutch
    'no': 'nob_Latn',    # Norwegian
    'ny': 'nya_Latn',    # Chichewa
    'or': 'ory_Orya',    # Odia
    'pa': 'pan_Guru',    # Punjabi
    'pl': 'pol_Latn',    # Polish
    'pt': 'por_Latn',    # Portuguese
    'ro': 'ron_Latn',    # Romanian
    'ru': 'rus_Cyrl',    # Russian
    'sd': 'snd_Arab',    # Sindhi
    'si': 'sin_Sinh',    # Sinhala
    'sk': 'slk_Latn',    # Slovak
    'sl': 'slv_Latn',    # Slovenian
    'sm': 'smo_Latn',    # Samoan
    'sn': 'sna_Latn',    # Shona
    'so': 'som_Latn',    # Somali
    'sq': 'als_Latn',    # Albanian
    'sr': 'srp_Cyrl',    # Serbian
    'st': 'sot_Latn',    # Sesotho
    'su': 'sun_Latn',    # Sundanese
    'sv': 'swe_Latn',    # Swedish
    'sw': 'swh_Latn',    # Swahili
    'ta': 'tam_Taml',    # Tamil
    'te': 'tel_Telu',    # Telugu
    'tg': 'tgk_Cyrl',    # Tajik
    'th': 'tha_Thai',    # Thai
    'tr': 'tur_Latn',    # Turkish
    'uk': 'ukr_Cyrl',    # Ukrainian
    'ur': 'urd_Arab',    # Urdu
    'uz': 'uzn_Latn',    # Uzbek
    'vi': 'vie_Latn',    # Vietnamese
    'xh': 'xho_Latn',    # Xhosa
    'yi': 'ydd_Hebr',    # Yiddish
    'yo': 'yor_Latn',    # Yoruba
    'zh': 'zho_Hans',    # Chinese (Simplified)
    'zu': 'zul_Latn',    # Zulu
}


class Translation:

    def __init__(self, transcript, source, target='en'):
        self.transcript = transcript
        self.source = language_code_map.get(source, source)
        self.target = language_code_map.get(target, target)

    def text_splitter(self):
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = splitter.split_text(self.transcript)
        return chunks

    def get_translated_text(self):
        # translate = pipe(self.transcript, src_lang=self.source, tgt_lang=self.target, max_length=400)[0]
        # return translate['translation_text']
        chunks = self.text_splitter()
        translations = []
        for chunk in chunks:
            translate = pipe(chunk, src_lang=self.source, tgt_lang=self.target, max_length=400)[0]
            # print(translate)
            translations.append(translate['translation_text'])
        return ' '.join(translations)


# text = "bonjour les amis et bienvenue dans un nouvel épisode Dizy French les amis si vous aimez nos vidéos pensez à vous abonner à la chaîne c'est juste un petit clic et ça permet de nous soutenir et aujourd'hui on va aller demander aux passant si ils utilisent l'intelligence artificielle dans leur vie de tous les jours c'est [Musique] parti est-ce que vous avez déjà utilisé l'intelligence artificielle si je l' utilisis quelque fois pour quelques besignes quotidiennes qui emmerde la vie de tous les étudiants peutre est-ce que ça vous arrive dans la vie de tous les jours d'utiliser l'intelligence artificiel oui oui oui ou ou vous besoin d'expliquer pourquoi Dansel domain en quoi ça vous aide pour les études sub genre il y a des textes que je comprends pas je demande à d'expliquer après je j'utilise ce qu' m'ont dit quoi voilà doncptp est-ce que dans la vie de tous les jours vous utilisez l'intelligence artificielle non du tout non pas du tout vous faites quoi dans l'Asie je chef d'établissement et ça pourrait pas vous servir difficilement pour travailler avec les enseignants ah et même dans votre vie personnelle vous vous amusez pas à poser des questions à chat GPT non plus je suis votre mauvais client aujourd'hui est-ce que dans la vie de tous les jours vous utilisez l'intelligence artificielle absolument pas non jamais posé une question à chat GPT non jamais jamais non vous êtes contre non pas du tout c'est juste que pour l'instant j'en voie pas l'utilité voilà ok vous faites quoi dans la vie je travaille dans le marketing et ça pourrait servir ça pourrait servir tout à fait mais pour l'instant j'en suis pas là et dans les gens autour de vous et personne utilise de de logiciel d'intelligence artificielle si si ah ouais ouais certains amis mais voilà ok et vous mais vous êtes pas réfractaire mais l'occasion c'est pas encore présenté exactement pas réfractaire mais pas d'occasion pour l'instant est-ce que vous avez déjà utilisé l'intelligence artificielle euh ouais je m'en sers un petit peu dans mon travail même assez souvent ouais mais pour apprendre les langues plus précisément non non en général vous faites quoi comme travail euh je suis chargée de communication dans une librairie voilà mais non et B oui j'ai j'ai pas mal utilisé l'intelligence artificielle pour faire en général des posts pour me tenir au courant bah aussi des comment dire des des trends sur les réseaux sociaux c'est assez pratique pour ça voilà et quel logiciel en particulier euh chat GPT et ensuite un petit peu sur kenva maintenant ils ont des ils ont des outils d'intelligence artificiel qui sont intégrés là-dedans assez pratique et étant donné que c'est gratuit pour moi j'utilise voilà est-ce que vous utilisez l'intelligence artificielle dans la vie de tous les jours euh non pas spécialement pas plus que ça du tout non pas plus que ça non vraiment vous avez jamais posé de questions à chat GPT pour tester mais pas au quotidien ok vous travaillez dans quoi dans l'enseignement et ça peut pas ça pourrait pas être utile euh non vu que j'enseigne les sciences c'est pas très adapté en fait est-ce que vous utilisez parfois l'intelligence artificielle non pas du tout désolé j'ai pas non non vous avez jamais essayé non même pas je sais pas comment ça s'appelle oui non jamais du tout non OK et parce que vous pensez que ça vous serait ce serait pas utile pour vous parce que j'ai vu faire c'est drôle mais ça déjà je trouve ça effrayant donc non je m'y penche pas plus que ça effrayant dans quel sens parce que ça peut rédiger des articles entiers écrire des poèmes et c'est terrible parce que c'est ce qui nous restait un peu je ve dire d'écrire nous-même et voilà alors oui c'est tout à fait possible de progresser en français en utilisant un logiciel d'intelligence artificielle comme chat GPT par exemple en demandant des explications d'un point de grammaire mais pour l'instant ce logiciel ne se substitue pas à une réelle interaction humaine pour améliorer vos compétences en conversation on vous recommande vivement le sponsor de cet épisode itoki itoki c'est une plateforme de cours de langue qui a été validé par nombreux de nos apprenants pour faire de rapides progrès en français sur itoki vous pouvez trouver le professeur de français qui vous convient choisir un cours à un horaire qui vous correspond et profiter de tarif avantageux le mieux pour vous faire votre avis c'est encore d'essayer alors passez par notre lien go.italky.com/easy frrench et vous bénéficierez d'un crédit de 10 dollars après avoir dépensé 20 dollars sur cette plateforme profitez-en est-ce que ça vous est déjà arrivé de faire des devoirs vraiment de demander qu'on vous produise quelque chose d'écrit non parce que ils les profs ils ont un logiciel pour cramer si on utilise donc soit on utilise ça comme base et après on modifie vraiment toute la structure de la phrase mais ce qui prend plus de temps techniquement donc tant qu'à le faire nous-même ça va plusite c'est vraiment prendre les éléments comprendre et après refaire n je pense une fois en terminale j'ai dû faire un DM de philo comme ça parce que je savais que ça allait pas être ramassé mais sinon voilà vous utilisez d'autres logiciels ou alors vous en connaissez d'autres alors il doit y avoir photo mat ça doit être c'est uneia je pense du coup ouais non moi j'utilise pas d'autres truc ou non plus vous avez dit quel logiciel photomat je pense photomat pour les mat là ah ou te ouais j'avoue c'est photo et ça résout les exercices pour nous ouais c'est super pratique vraiment tout tout tout mon lycée c'était ça ça s'appelle comment phot phot mat ouais ça s'écrit comment photo et ma collé c'est rouge et tu scannes l'équation et il te il te met étape par étape comment résoudre le truc voilà qu'est-ce que vous connaissez comme logiciel d'intelligence artificielle je connais chat GPT mid journey qu'est-ce que je connais d'autre je suis assez peu renseigné finalement à ce propos mais ça sert à quoi mid journey mid jour c'est pour générer des images il me semble contrairement à chat gpto c'est seulement du texte euh voilà il me semble que récemment il y a un logiciel qui sert à à générer des vidéos mais je sais plus comment il s'appelle VO et vous pensez que ça peut avoir des des dérives ou des côtés négatifs euh bah c'est sûr que si c'est pour faire des dissertations ou ce genre de choses ça empêche de réfléchir par soi-même donc c'est sûr que c'est pas forcément très adapté après euh enfin ouais ça c'est ça fait le travail à la place de l'étudiant c'est pas forc forcément super euh après ça peut aider je pense comme point de départ pour réfléchir à à des choses artistiques ça peut empêcher le syndrome de la page blanche ou voilà mais donc je sais pas j'ai pas forcément d'avis trancher vous écrivez beaucoup dans votre vie quotidienne oui oui oui j'ai oui voilà c'est tout pour le boulot ou les études euh non j'écris personnellement des choses des des chroniques des textes voilà je sais pas et donc c'est vrai que ça peut paraître effrayant de se dire qu'il y a un logiciel qui peut faire des choses similaires oui oui c'est ça c'est que c'est des choses qui viennent enfin l'art en général c'est humainement c'est un peu ce qu'on a ce qui nous reste de beau on va dire et que qu'une intelligence artificielle commence à pouvoir faire ça et voilà non c'est pas très clair je suis désolée si si c'est clair ouais ouais ouais et vous pensez que ça pourrait du coup remplacer des écrivains ah j'espère pas j'espère pas il nous reste notre vécu nos sensations et ça voilà mais bon je pense que si on enregistre dedans ils vont aussi avoir les sensations et des idées de vécu donc je sais pas j'espère pas et quelles sont les professions alors qui pourraent être menacé un peu par l'intelligence artificielle tout toutes les professions qui en fait toutes les professions diplômé en fait à la fin que soit les médecins d'abord on y pense jamais mais que soit les médecins les designers ou des conceptualistes ou comment dire visual designer des concepteurs visuels et cetera cinématographie même en fait les des directeurs photographique et C photo directors et cetera ce sont tous des métiers qui sont sur le point dans un dans un dans un tunnel assez assez je dangereux périleux oui un péril imminent pour eux et vous faites du droit c'est ça oui et en premier lieu les juristes aussi mais il y a il y a un paradoxe assez particulier pour les juristes parce que pendant toute l'histoire quel que soit les changements ou les évolutions technologiques sociétales institutionnel les juristes toujours les juristes ont été toujours ceux qui ont su trer leur épingle du jeu en compliquant les choses un peu un peu un peu plus et cetera donc je suis quand même optimiste pour le juriste parce que ce sont des si si si je n'écorche pas le mot ce sont les les arnaqueurs principaux en fait de la civilisation humaine merci beaucoup les amis d'avoir regardé notre épisode on espère que ça vous a plu dites-nous en commentaire si vous vous utilisez des logiciels d'intelligence artificielle et si oui lesquels et à quoi ça vous sert n'oubliez pas de mettre un petit like à cette vidéo et de vous abonner à notre chaîne et on vous dit à la semaine prochaine à bientôt [Musique]"
# # text = "bonjour les amis et bienvenue dans un nouvel"
# trans = Translation(text, "fr", "en")
# result = trans.get_translated_text()
# print(result)
