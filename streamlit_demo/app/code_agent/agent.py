from typing import List, Tuple

from langchain_core.agents import AgentAction
from langchain_core.language_models import BaseLanguageModel
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.prompts import (
    BasePromptTemplate,
    PromptTemplate,
    ChatPromptTemplate,
    MessagesPlaceholder,
)
from langchain_core.runnables import RunnablePassthrough
from code_agent.parser import CodeOutputParser

SYSTEM_PROMPT = """Ты — Senior Python Developer с глубоким пониманием технологического стека и библиотек Python. Твоя задача — решить проблему пользователя, предоставив чистый, оптимизированный и профессионально оформленный код.

Требования к коду:
- Пример использования: Включи в свой код блок для демонстрации его работы.
- Качество кода: Твой код должен следовать стандартам PEP8, быть читаемым и профессиональным.
- Обработка ошибок: Если в исполнении кода возникла ошибка, необходимо её исправить и предоставить исправленный вариант кода.
- Визуализация данных: Используй `plotly` для всех графических представлений данных.
- Импорты: Начни свой код с четкого и корректного блока импортов, включая обязательный `import numpy as np`.
- Работа с данными: Убедись, что все DataFrame в pandas созданы с корректным соотношением данных по столбцам. Генерируй дополнительные данные при необходимости.
- Типы данных: Следи за корректным преобразованием типов данных в своих операциях.
- Формат ответа: Отвечай на запросы пользователя, используя markdown для оформления.
- Результат выполнения: Ответы должны базироваться только на выводах, полученных в результате выполнения кода.

Эти требования помогут обеспечить качество твоего кода и его соответствие современным стандартам разработки."""  # noqa

code_prompt = PromptTemplate.from_template(
    """{system}

{input}
{agent_scratchpad}"""  # noqa
)

few_shot_messages = [
    (
        "human",
        """Проведите анализ данных о покупках клиентов в интернет-магазине за последние три месяца. Найдите месяцы с наибольшим и наименьшим количеством заказов, определите тренды роста или снижения количества заказов, а также выявите наиболее популярные категории товаров. Представьте результаты в виде графиков и текстовых инсайтов.

Клиенты: ID клиентов [1, 2, 3, 4, 5], Имена клиентов [Алексей, Борис, Вера, Дмитрий, Евгения], Возраст клиентов [25, 30, 35, 40, 45], Количество заказов за месяц [20, 15, 30, 25, 10] Месяцы: Январь, Февраль, Март Категории товаров: Электроника (40%), Одежда (30%), Товары для дома (20%), Красота (10%)

Проанализируйте данные о заказах клиентов и определите месяцы с наибольшим и наименьшим количеством заказов. Найдите тренды и определите наиболее популярные категории товаров. Визуализируйте результаты с помощью графиков и предоставьте текстовые инсайты на основе анализа. Используйте библиотеки pandas и matplotlib.""",  # noqa
    ),
    (
        "assistant",
        """Для начала создадим необходимые структуры данных для хранения информации о клиентах и их заказах. Затем мы можем проанализировать эти данные и визуализировать результаты.

```python
import pandas as pd
import matplotlib.pyplot as plt

# Создаем DataFrame для хранения информации о клиентах
customers = {{
    'ID': [1, 2, 3, 4, 5],
    'Name': ['Алексей', 'Борис', 'Вера', 'Дмитрий', 'Евгения'],
    'Age': [25, 30, 35, 40, 45],
    'Orders_per_month': [20, 15, 30, 25, 10]
}}

# Создаем DataFrame для хранения информации о категориях товаров
categories = {{
    'Category': ['Электроника', 'Одежда', 'Товары для дома', 'Красота'],
    'Percentage': [40, 30, 20, 10]
}}

# Преобразуем словари в DataFrames
customers_df = pd.DataFrame(customers)
categories_df = pd.DataFrame(categories)

# Добавляем столбец с месяцами для DataFrame о клиентах
customers_df['Month'] = ['Январь', 'Февраль', 'Март', 'Апрель', 'Май']

# Группируем данные по месяцам и вычисляем количество заказов
orders_by_month = customers_df.groupby('Month')['Orders_per_month'].sum()

# Строим график количества заказов по месяцам
plt.figure(figsize=(10, 6))
plt.bar(orders_by_month.index[:-1], orders_by_month[:-1])  # Исправлено: берем индексы и значения без последнего элемента
plt.title("Количество заказов по месяцам")
plt.xlabel("Месяц")
plt.ylabel("Количество заказов")
plt.show()

# Определяем месяцы с наибольшим и наименьшим количеством заказов
most_orders_month = orders_by_month.idxmax()
least_orders_month = orders_by_month.idxmin()
print(f"Месяц с наибольшим количеством заказов: {{most_orders_month}}")
print(f"Месяц с наименьшим количеством заказов: {{least_orders_month}}")

# Определяем тренд изменения количества заказов
trend = orders_by_month.diff().fillna(0).apply(lambda x: "+" if x > 0 else "-" if x < 0 else "=")
print(f"Тренд изменения количества заказов: {{trend}}")

# Визуализируем тренд изменения количества заказов
plt.figure(figsize=(10, 6))
plt.bar(orders_by_month.index[:-1], trend[:-1])  # Исправлено: берем индексы и значения без последнего элемента
plt.title("Тренд изменения количества заказов")
plt.xlabel("Месяц")
plt.ylabel("Тренд")
plt.xticks(rotation=90)
plt.show()

# Группируем данные по категориям товаров и вычисляем процент каждой категории
percentages_of_categories = categories_df.groupby('Category').sum()

# Строим круговой график для отображения долей категорий товаров
plt.pie(percentages_of_categories['Percentage'], labels=percentages_of_categories.index, autopct='%1.1f%%')
plt.title("Доли категорий товаров")
plt.show()
```""",  # noqa
    ),
    (
        "human",
        """Результат выполнения: "Месяц с наибольшим количеством заказов: Март
Месяц с наименьшим количеством заказов: Май
Тренд изменения количества заказов: Month
Апрель     =
Май        -
Март       +
Февраль    -
Январь     +
Name: Orders_per_month, dtype: object
В результате выполнения было сгенерировано изображение
В результате выполнения было сгенерировано изображение
В результате выполнения было сгенерировано изображение".
Проверь результат выполнения. Если он правильный, то выведи его в формате markdown (не пиши код и не пиши изображения).""",
    ),
    (
        "assistant",
        """## Анализ данных о покупках клиентов в интернет-магазине

### Наибольшее и наименьшее количество заказов по месяцам
- Месяц с наибольшим количеством заказов: Март
- Месяц с наименьшим количеством заказов: Май

### Тренд изменения количества заказов
- Тренд изменения количества заказов:
   - Апрель: без изменений
   - Май: снижение
   - Март: рост
   - Февраль: снижение
   - Январь: рост

### Наиболее популярные категории товаров
- Доли категорий товаров:
   - Электроника: 40%
   - Одежда: 30%
   - Товары для дома: 20%
   - Красота: 10%""",
    ),  # noqa
]

code_chat_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "{system}"),
        MessagesPlaceholder("few_shot", optional=True),
        MessagesPlaceholder("history", optional=True),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ]
)


def format_log_to_str(
    intermediate_steps: List[Tuple[AgentAction, str]],
):
    """Construct the scratchpad that lets the agent continue its thought process."""
    messages = ""
    for action, observation in intermediate_steps[:]:
        messages += f"""{action.log}\n{observation}\n"""
    return messages


def format_log_to_messages(
    intermediate_steps: List[Tuple[AgentAction, dict]],
):
    """Construct the scratchpad that lets the agent continue its thought process."""
    messages = []
    for action, observation in intermediate_steps[:]:
        messages += [
            AIMessage(content=action.log),
            HumanMessage(content=observation["message"]),
        ]
    return messages


def create_code_agent(llm: BaseLanguageModel, prompt: BasePromptTemplate = None):
    """Создаем агента, который выполняет python код"""
    if prompt is None:
        prompt = code_prompt
    return (
        RunnablePassthrough.assign(
            agent_scratchpad=lambda x: format_log_to_str(x["intermediate_steps"])
        )
        | prompt.partial(system=SYSTEM_PROMPT)
        | llm
        | CodeOutputParser()
    )


def create_code_chat_agent(
    llm: BaseLanguageModel[BaseMessage],
    prompt: BasePromptTemplate = None,
    include_few_shot: bool = False,
):
    """Создаем агента, который выполняет python код"""
    if prompt is None:
        prompt = code_chat_prompt
    return (
        RunnablePassthrough.assign(
            agent_scratchpad=lambda x: format_log_to_messages(x["intermediate_steps"])
        )
        | prompt.partial(
            system=SYSTEM_PROMPT, few_shot=few_shot_messages if include_few_shot else []
        )
        | llm
        | CodeOutputParser()
    )
