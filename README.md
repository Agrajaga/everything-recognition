# everything-recognition
Проект демонстрирует работу с функцией поиска объектов на изображении, входящей в библиотеку машинного зрения [OpenCV](https://github.com/opencv/opencv). 

Поиск объектов основан на методе Виолы-Джонса. В основе метода лежит интегральное представление изображения по признакам Хаара, построение классификатора на основе алгоритма адаптивного бустинга и комбинирование классификаторов в каскадную структуру. Алгоритм при поиске работает очень быстро и может быть использован для обнаружения лиц и других объектов в режиме реального времени. Однако у него есть и недостатки — при наклоне лица на угол более 30 градусов вероятность обнаружения резко падает.

## Установка
Для работы необходима установленная версия `python3`. Для добавления требуемых зависимостей используйте `pip` (или `pip3` при конфликтах с `python2`) - выполните команду:
```
pip install -r requrements.txt
```

## Использование
Для запуска введите:
```
python run.py
```
В результате произойдет подключение к вашей Web-камере и в режиме реального времени в окне на экране будет отображаться изображение с камеры, а на нем будут выделяться цветной рамкой найденные объекты: лица (в фас и профиль) и глаза.
Для выхода из программы необходимо нажать клавишу `q` на клавиатуре.


## Техническое описание
В файле `run.py` используются следующие функции:

__`get_cascades`__  
Создает и настраивает список классификаторов для поиска объектов.  
Возвращает список объектов `cv2.CascadeClassifier`.

__`show_frame`__  
Выводит на экран обработанные кадры.  
Принимает на вход матрицу `numpy.ndarray`, содержащую обработанный текущий кадр с web-камеры.

__`draw_sqare`__  
Добавляет на текущий кадр рамку вокруг области, содержащей найденный объект.  
Принимает на вход матрицу `numpy.ndarray` текущего кадра и цвет рамки `tuple[Blue, Green, Red]`. Координаты рамки задаются через глобальный контекст.

__`is_user_wants_quit`__  
Функция отслеживает ввод пользователя с клавиатуры и если нажатая клавиша соответствует `q`, то возвращает `True`.

Какие именно объекты будут определяться программой задается в файле `config.py`. Он содержит словарь с описаниями для классификаторов, каждое описание содержит реквизты:
- `path` - `str` путь к файлу, содержащему предобученный классификатор для данного типа объектов;
- `color` - `tuple[Blue, Green, Red]` цвет рамки вокруг найденного объекта;
- `draw` - `bool` флаг, указывающий выделять данный объект или нет.

Предобученные каскадные классификаторы Хаара находятся в папке `haarcascades`.

