import pymorphy2
from sklearn.feature_extraction.text import CountVectorizer
import torch
import torch.nn as nn
import numpy as np
import string

def drop_punctuation(line, sym):
  for p in sym:
    if p in line:
        line = line.replace(p, '')
  return line

morph = pymorphy2.MorphAnalyzer()
def normilise_text(text, morph):
  return ' '.join([morph.parse(word)[0].normal_form for word in text.split() if word != ''])

symbols_to_avoid = string.punctuation
symbols_to_avoid += '—«»“”‘’$€£¥₽'
categories = ['Авиатранспорт','Автосервисы','Арбитражные слуги, третейские суды','Техника: аудио-, видео-, фото-, кино-','Аудио, видеопродукция, игры','Биологически активные добавки','Благотворительная деятельность','Бытовая техника','Бытовая химия','Водный транспорт','Грузовые автомобили','Досуг и развлечения','Компьютерная техника и ПО','Косметика, уход, гигиена','Легковые автомобили и мототехника','Лекарственные препараты','Массовые мероприятия','Мебель и предметы интерьера','Медицинское оборудование и материалы','Напитки','Общественные организации','Одежда и обувь','Оргтехника и канцелярские товары','Отопительное и водонагревательное оборудование','Парфюмерия','Политическая реклама','Посредничество и франчайзинг','Продукты питания','Промышленное оборудование','Промышленные материалы','Реклама социальная','Сертификация','Изделие, конструктивно схожее с оружием','Спортивные изделия','Средства массовой информации','Средства охраны и личной безопасности','Средства связи и оборудование','Строительные, отделочные материалы, сантехника','Текстиль','Товары и техника для сада','Товары для детей','Товары для животных','Транспорт и ГСМ','Туризм, спорт и отдых','Услуги бытовые и сервис','Услуги в области интернета','Услуги в области рекламы и маркетинга','Услуги в области торговли','Услуги в системе образования и трудоустройство','Услуги медицинские','Услуги общественного питания','Услуги по операциям с недвижимостью','Услуги по охране и безопасности','Услуги производственные','Услуги связи','Услуги страховые','Услуги транспортные','Услуги финансовые','Услуги юридические, аудиторские и консалтинговые','Хозяйственные принадлежности (механические)','Часы, ювелирные изделия','Музыкальные инструменты','Азартные игры','Наркотические и психотропные вещества','Оружие','Сырье для предпр. легкой промышленности','Нецензурные выражения, оскорбления, негатив','Порнография и эротика','Шпионаж','Логотипы и бренды']

category_encoded_map = dict()
for i, c in enumerate(categories):
  category_encoded_map[c] = i

category_encoded_map_reversed = dict()
for category, category_id in category_encoded_map.items():
  category_encoded_map_reversed[category_id] = category

class My_model(nn.Module):
    def __init__(self, size_in, size_hide, size_out):
        super().__init__()
        self.linear_1 = nn.Linear(size_in, size_hide)
        self.bn = nn.BatchNorm1d(size_hide)
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()
        self.dp = nn.Dropout(0.8)
        self.linear_2 = nn.Linear(size_hide, size_out)

    def forward(self, x):
        x = x.to(torch.float32)
        x = self.linear_1(x)
        x = self.bn(x)
        x = self.tanh(x)
        # x = self.relu(x)
        x = self.dp(x)
        x = self.linear_2(x)
        return x

    def predict(self, input, drop_punctuation, normilise_text, vectorizer, device='cpu'):
        clear_text = drop_punctuation(input, symbols_to_avoid)
        tokens_text = normilise_text(clear_text, morph)
        text_vector = vectorizer.transform([tokens_text]).toarray()[0]

        output = self.forward(torch.tensor([text_vector], dtype=torch.int64)).detach().numpy()

        category_id = np.argmax(output)
        return category_encoded_map_reversed[category_id]
