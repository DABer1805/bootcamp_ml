import streamlit as st
import numpy as np
import joblib
import pickle

# Значения для признаков с выбором оценки
MARKS: tuple[str, str, str, str, str] = ('5', '4', '3', '2', '1')


class Comment:
    """ Комментарий """

    def __init__(
            self,
            status: int,
             text: str,
            author_name: str
    ) -> None:
        # Тип оценки: негативная или позитивная
        self.status: int = status
        # Текст комментария
        if text:
            self.text: str = text
        else:
            self.text: str = 'Пустой отзыв'
        # Имя автора
        if author_name:
            self.author_name: str = author_name
        else:
            self.author_name: str = 'Пользователь'


def write_comment(comment: Comment) -> None:
    """ Отрисовываем комментарий пользователя с именем автора и оценкой
    отзыва

    """
    # Выбираем оценку отзыва в соответствии с предсказанием модели
    if comment.status:
        status: str = '🟩 ***Положительный отзыв:***'
    else:
        status: str = '🟥 ***Отрицательный отзыв:***'

    # Отрисовываем комментарий
    st.write(f'### {status} 👤 *{comment.author_name}*')
    st.write(f'✈️ {comment.text}')
    st.write('---')


def predict_status(features: np.ndarray, comment_text: str, author_name: str):
    """ Предсказать оценку отзыва и отрисовать комментарий пользователя """
    # Загружаем скейлер и модель
    scaler = joblib.load('scaler.pkl')
    with open('model.pickle', 'rb') as f:
        model = pickle.load(f)

    # Отрисовываем комментарий пользователя
    #write_comment(
    #    Comment(
    #        model.predict(scaler.transform(features)).round().astype(int),
    #        comment_text, author_name
    #    )
    #)


def start() -> None:
    """ Отрисовка страницы и боковой панели """
    # Задаём параметры страницы
    st.set_page_config(
        layout="wide",
        initial_sidebar_state="auto",
        page_title="ML Airlines",
    )
    # Отрисовка заголовка
    draw_header()
    # Отрисовка боковой панели
    draw_sidebar()


def draw_header() -> None:
    """ Отрисовка заголовка """
    st.write(
        '''
        # 🛸 ***Отзывы клиентов компании "ML Airlines"***
        ### 🖥 Определение нейросетью - доволен клиент авивперелётом или нет

        ---

        '''
    )


def draw_sidebar() -> None:
    """ Отрисовка боковой панели """
    st.sidebar.header('Заданные пользователем параметры')
    # Имя пользователя
    user_name = st.sidebar.text_input('Введите ваше имя')

    # Лоялен ли пользователь к авиакомпании
    user_loyalty: int = int(
        st.sidebar.checkbox('Лояльны ли вы к нашей компании?')
    )

    # Тип поездки пользователя
    user_travel_type: str = st.sidebar.selectbox(
        'Каков ваш тип поездки?', ('Business', 'Personal')
    )

    # Класс обслуживания пользователя
    user_service_class: str = st.sidebar.selectbox(
        'Класс обслуживания', ('Business', 'Eco', 'Eco Plus')
    )

    # Дальность полёта пользователя
    user_flight_distance: int = int(st.sidebar.slider(
        'Дальность вашего полёта (в милях)', min_value=200,
        max_value=10000, value=200, step=1
    ))

    # Оценка пользователем интернета
    inflight_wifi_service: str = st.sidebar.selectbox(
        'Оцените качество интернета', MARKS
    )

    # Оценка пользователем удобства покупки билетов
    ticket_purchase_quality: str = st.sidebar.selectbox(
        'Оцените удобство покупки билетов', MARKS
    )

    # Оценка пользователем удобства посадки в самолёт
    the_quality_of_boarding_the_plane: str = st.sidebar.selectbox(
        'Оцените удобство посадки в самолёт', MARKS
    )

    # Оценка пользователем качества еды и напитков
    the_drink_and_food_quality: str = st.sidebar.selectbox(
        'Оцените качество еды и напитков на борту', MARKS
    )

    # Оценка пользователем удобства сидений
    seat_quality: str = st.sidebar.selectbox(
        'Оцените удобство сидений на борту', MARKS
    )

    # Оценка пользователем качества развлечений
    quality_of_entertainment: str = st.sidebar.selectbox(
        'Оцените качество развелечений', MARKS
    )

    # Оценка пользователем качества обслуживания на борту
    quality_of_service_on_board: str = st.sidebar.selectbox(
        'Оцените качество обслуживания на борту', MARKS
    )

    # Оценка пользователем того, как удобно было ногам
    leg_room_service_quality: str = st.sidebar.selectbox(
        'Оцените, насколько удобно было вашим ногам', MARKS
    )

    # Оценка пользователем качества обращения с багажом
    baggage_handling_quality: str = st.sidebar.selectbox(
        'Оцените качество обращения с багажом', MARKS
    )

    # Оценка пользователем чистоты на борту
    cleanliness_quality: str = st.sidebar.selectbox(
        'Оцените, как чисто было на борту', MARKS
    )

    # Оценка пользователем качества регистрации
    checkin_service_quality: str = st.sidebar.selectbox(
        'Оцените качество регистрации', MARKS
    )

    # Комментарий пользователя (эмоциальный прогноз по тексту комментария
    # НЕ ПРОВОДИТСЯ. Оценка довольности клиента проводится исключительно
    # по показателям выше. Данный элемент нужен для полной связки:
    # эмоциальная оценка отзыва -> автор -> текст отзыва)
    user_comment: str = st.sidebar.text_area('Ваш комментарий')
    # Кнопка с отправкой отзыва -> предсказанием модели
    if st.sidebar.button('Оставить отзыв'):
        predict_status(np.array(
            [
                [
                    user_loyalty,
                    *get_binary_features(user_travel_type, ['Business']),
                    user_flight_distance,
                    *get_binary_features(
                        user_service_class, ['Business', 'Eco']
                    ),
                    *get_binary_features(
                        inflight_wifi_service, ['2', '3', '4', '5']
                    ),
                    *get_binary_features(
                        ticket_purchase_quality, ['5']
                    ),
                    *get_binary_features(
                        the_drink_and_food_quality, ['1']
                    ),
                    *get_binary_features(
                        the_quality_of_boarding_the_plane,
                        ['1', '2', '3', '4', '5']
                    ),
                    *get_binary_features(
                        seat_quality,
                        ['1', '2', '3', '4', '5']
                    ),
                    *get_binary_features(
                        quality_of_entertainment,
                        ['1', '2', '3', '4', '5']
                    ),
                    *get_binary_features(
                        quality_of_service_on_board,
                        ['1', '2', '4', '5']
                    ),
                    *get_binary_features(
                        leg_room_service_quality,
                        ['1', '2', '3', '4', '5']
                    ),
                    *get_binary_features(
                        baggage_handling_quality,
                        ['3', '5']
                    ),
                    *get_binary_features(
                        checkin_service_quality,
                        ['1', '2', '5']
                    ),
                    *get_binary_features(
                        quality_of_service_on_board,
                        ['3', '5']
                    ),
                    *get_binary_features(
                        cleanliness_quality,
                        ['1', '2', '5']
                    ),
                ]
            ]
        ), user_comment, user_name
        )


def get_binary_features(value: str, variants: list[str]) -> list[int]:
    """ Сопоставляем выбранное значение со списком на совпадение """
    return list(
        map(
            lambda x: value == x, variants
        )
    )


if __name__ == "__main__":
    start()
