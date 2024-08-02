from langchain.prompts import ChatPromptTemplate, PromptTemplate, FewShotChatMessagePromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from vllm import LLM, SamplingParams
from langchain_community.llms import Ollama
from langchain.schema.messages import HumanMessage, SystemMessage
from langchain.prompts import ChatPromptTemplate
import sys
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field, validator
from typing import List, Optional
from langchain.prompts import ChatPromptTemplate, PromptTemplate, FewShotChatMessagePromptTemplate
from transformers import AutoTokenizer, PreTrainedTokenizerFast
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
import json
import requests
import os
import re
import time
from utils import model_map, get_format_instructions, ChatChain, str_to_json

system_message = "Ты профессиональный экзаменатор с глубоким знанием предмета. Ты должен отвечать на Русском языке."

class Case(BaseModel):
    case_desc: str = Field(description="Описание кейса")
    
class GeneratedQuestion(BaseModel):
    questions: list[str] = Field(description="Сгенерированные вопросы")


parser_case = PydanticOutputParser(pydantic_object=Case)
parser_generation = PydanticOutputParser(pydantic_object=GeneratedQuestion)

generation_format_instructions = 'Выведи результат в блоке json следуя следующему формату:\n```json\n{\n\"questions\": [\n\t{\n\t\t"competence": "<компетенция 1>",\n\t\t"question": "<вопрос для компетенции 1>"\n\t},\n\t{\n\t\t"competence": "<компетенция 2>",\n\t\t"question": "<вопрос для компетенции 2>"\n\t},\n\t...\n\t]\n}\n```'

few_shot_examples = {
    'case': [
        {
            'competence_list': ' - Анализирует и интерпретирует результаты работы моделей машинного обучения\n - Разрабатывает и тестирует алгоритмы для обработки больших данных\n - Оценивает и минимизирует риски при использовании искусственного интеллекта\n - Осуществляет контроль качества данных и процессов обучения моделей\n - Использует методы объяснимого ИИ для интерпретации моделей',
            'output': 'Анализ компетенций:\n\n1. **Анализирует и интерпретирует результаты работы моделей машинного обучения**:\n   - Эта компетенция предполагает умение понимать и объяснять результаты работы моделей, в том числе сложных нейронных сетей и других алгоритмов. Студент должен быть способен объяснить, почему модель выдает определенные прогнозы или классификации.\n   \n2. **Разрабатывает и тестирует алгоритмы для обработки больших данных**:\n   - Эта компетенция требует от студента умения создавать и тестировать алгоритмы, которые способны эффективно обрабатывать большие объемы данных. Это включает в себя оптимизацию процессов, работу с распределенными системами и выбор подходящих алгоритмов для конкретных задач.\n   \n3. **Оценивает и минимизирует риски при использовании искусственного интеллекта**:\n   - Данная компетенция фокусируется на идентификации и управлении потенциальными рисками, связанными с использованием ИИ. Студент должен понимать, как ИИ может повлиять на организацию или общество, и разрабатывать стратегии для минимизации этих рисков.\n   \n4. **Осуществляет контроль качества данных и процессов обучения моделей**:\n   - Это включает в себя проверку и обеспечение качества данных перед их использованием в моделях машинного обучения. Студент должен быть способен разрабатывать и применять процедуры, которые гарантируют, что данные корректны, полны и не содержат ошибок.\n   \n5. **Использует методы объяснимого ИИ для интерпретации моделей**:\n   - Эта компетенция связана с умением применять инструменты и методы для объяснения работы сложных моделей ИИ. Студент должен быть способен сделать модели прозрачными и понятными для пользователей, включая не технических специалистов.\n\nНа основе анализа этих компетенций можно сделать вывод, что кейс должен быть связан с применением ИИ для решения сложной задачи, включающей работу с большими данными, где важна интерпретация и минимизация рисков.\n\nКейс:\n\n```json\n{\n  \"case_desc\": \"Вы - ведущий специалист в команде разработки системы поддержки принятия решений в области здравоохранения. Система разрабатывается для крупной сети клиник и предназначена для анализа и прогнозирования исходов лечения пациентов на основе данных электронной медицинской карты, медицинских изображений, лабораторных анализов и генетической информации. Ключевая цель системы - помочь врачам принимать обоснованные решения о назначении терапии, прогнозировании возможных осложнений и оптимизации лечебного процесса. Ваша задача - разработать и внедрить алгоритмы для обработки и анализа больших объемов данных, интерпретировать результаты работы модели и обеспечить контроль качества данных. Помимо этого, важно учитывать этические аспекты, связанные с использованием персональных медицинских данных, и минимизировать риски, связанные с возможными ошибками системы, которые могут повлиять на здоровье пациентов. Также необходимо использовать методы объяснимого ИИ для того, чтобы врачи могли понимать, на чем основаны прогнозы и рекомендации системы, и уверенно принимать решения.\"\n}\n```',
            'format_instructions': get_format_instructions(parser_case)
        },
        {
            'competence_list': ' - Применяет математический аппарат для решения задач по оценке и разработке моделей\n - Разрабатывает и реализует архитектуру ансамбля моделей\n - Использует программные и технические средства для визуализации больших данных\n - Решает задачи искусственного интеллекта (ИИ)\n - Разрабатывает и применяет методы машинного обучения (МО) для решения задач\n - Применяет нормы этики при использования данных',
            'output': 'Анализ компетенций:\n\n1. **Применяет математический аппарат для решения задач по оценке и разработке моделей**:\n   - Эта компетенция предполагает использование математических методов и статистических подходов для оценки эффективности и разработки моделей машинного обучения. Студент должен владеть инструментами математического анализа, регрессии, вероятностных моделей и оптимизации.\n\n2. **Разрабатывает и реализует архитектуру ансамбля моделей**:\n   - Эта компетенция требует знания и опыта в создании ансамблевых моделей, которые объединяют несколько моделей для улучшения прогнозирования. Студент должен уметь выбирать и комбинировать различные модели, такие как случайные леса, градиентный бустинг и ансамбли нейронных сетей.\n\n3. **Использует программные и технические средства для визуализации больших данных**:\n   - Важный аспект работы с большими данными — визуализация результатов. Компетенция требует умения использовать инструменты для визуализации, такие как Python-библиотеки (например, Matplotlib, Seaborn), а также специализированные платформы для работы с большими данными (например, Tableau, Power BI).\n\n4. **Решает задачи искусственного интеллекта (ИИ)**:\n   - Эта компетенция предполагает умение решать различные задачи ИИ, включая классификацию, регрессию, кластеризацию и обработку естественного языка. Студент должен быть в состоянии применять алгоритмы ИИ в различных прикладных задачах.\n\n5. **Разрабатывает и применяет методы машинного обучения (МО) для решения задач**:\n   - Компетенция включает в себя разработку и внедрение методов МО для решения конкретных задач, таких как предсказание, классификация, выявление аномалий и т.д. Это требует навыков программирования, понимания алгоритмов и опыта работы с различными типами данных.\n\n6. **Применяет нормы этики при использования данных**:\n   - Этические нормы играют ключевую роль при работе с данными, особенно если это касается персональных данных. Студент должен понимать важность конфиденциальности, справедливости и прозрачности при использовании данных и внедрении ИИ.\n\nНа основе анализа компетенций, кейс должен быть сосредоточен на реальной задаче, которая включает обработку больших данных, разработку моделей машинного обучения и ИИ, с особым вниманием к этическим аспектам использования данных.\n\nКейс:\n\n```json\n{\n  \"case_desc\": \"Вы являетесь руководителем команды разработчиков в крупной IT-компании, которая работает над созданием системы рекомендаций для онлайн-платформы электронной коммерции. Цель системы - предоставление персонализированных рекомендаций пользователям на основе их предыдущих покупок, поведения на сайте, и анализа данных из социальных сетей. Ваша задача включает в себя сбор и подготовку данных, разработку архитектуры ансамбля моделей, которые будут сочетать рекомендации на основе содержания и коллаборативную фильтрацию, а также визуализацию результатов работы модели для представления их бизнес-партнерам. Важно учитывать нормы этики при использовании персональных данных пользователей, включая вопросы конфиденциальности и справедливости рекомендаций. Необходимо также разработать систему отчетности, которая покажет, как модель принимает решения, чтобы обеспечить доверие пользователей и соответствие этическим стандартам.\"\n}\n```',
            'format_instructions': get_format_instructions(parser_case)
        }
    ],
    'generation': [
        {
            'case_desc': 'В настоящее время сотрудники многих компаний и организаций подвержены переменчивым настроениям и склонны к частой смене места работы. Деятельность значительного числа высококвалифицированных сотрудников так или иначе связана с использованием современных ИТ-инструментов коммуникации и профессиональной деятельности, в том числе личных средств интернет-коммуникации (блоги, социальные сети, в т.ч., профессиональные), что дает возможность анализа настроений сотрудников по поводу смены места работы.\n\nВы являетесь руководителем проекта по разработке и внедрению интеллектуальной системы предиктивной аналитики увольнения сотрудников (ИС ПАУС), задача первой версии которой - раннее обнаружение настроений сотрудников к смене места работы и оповещение об этом ответственных лиц. В компании работает 10-15 тыс. сотрудников. Бюджет проекта - 60 млн. руб. на 1 год. Предполагается, что высококвалифицированные сотрудники работают в офисе (либо удалённо, но с использованием VPN компании для удалённого доступа к информационным ресурсам компании), используют для коммуникации корпоративную электронную почту, корпоративный мессенджер, а также ведут личные страницы в социальных сетях. У потенциальной системы нет жестких ограничений по обеспечению ее функционирования в реальном времени.',
            'competence_list': ' - Применяет математический аппарат для решения задач по оценке и разработке моделей\n - Разрабатывает и реализует архитектуру ансамбля моделей\n - Использует программные и технические средства для визуализации больших данных\n - Решает задачи искусственного интеллекта (ИИ)\n - Разрабатывает и применяет методы машинного обучения (МО) для решения задач\n - Применяет нормы этики при использования данных',
            'output': 'Для того чтобы создать подходящие вопросы, сначала нужно тщательно проанализировать кейс и компетенции.\n\n### Анализ компетенций и кейса\n\n**Кейс:** \nВы разрабатываете интеллектуальную систему предиктивной аналитики для выявления настроений сотрудников, связанных с возможным увольнением. Эта система будет собирать и анализировать данные из различных источников, таких как корпоративные и личные средства коммуникации. Основная задача — прогнозирование риска увольнения и своевременное оповещение ответственных лиц. \n\n**Компетенции:** \n1. **Применяет математический аппарат для решения задач по оценке и разработке моделей**: Умение использовать математические методы для разработки и оценки точности моделей, что критически важно для создания предиктивных систем.\n2. **Разрабатывает и реализует архитектуру ансамбля моделей**: Компетенция связана с созданием сложных моделей, объединяющих результаты нескольких моделей для улучшения прогнозирования.\n3. **Использует программные и технические средства для визуализации больших данных**: Необходимо визуализировать данные и результаты модели для лучшего понимания и анализа.\n4. **Решает задачи искусственного интеллекта (ИИ)**: Включает разработку и использование ИИ для решения сложных задач, таких как анализ настроений.\n5. **Разрабатывает и применяет методы машинного обучения (МО) для решения задач**: Создание моделей, которые могут обучаться и прогнозировать на основе данных.\n6. **Применяет нормы этики при использования данных**: Важно соблюдать этические нормы, особенно при работе с персональными данными сотрудников.\n\n### Генерация вопросов\n\nВопросы должны быть направлены на оценку каждой компетенции в контексте описанного кейса.\n\n```json\n{\n\"questions\": [\n\t{\n\t\t"competence": "Применяет математический аппарат для решения задач по оценке и разработке моделей",\n\t\t"question": "Какой математический подход вы бы выбрали для оценки точности модели предсказания настроений сотрудников к увольнению? Какие метрики вы считаете наиболее подходящими для этого случая и почему?"\n\t},\n\t{\n\t\t"competence": "Разрабатывает и реализует архитектуру ансамбля моделей",\n\t\t"question": "Вы решили использовать ансамблевый метод для улучшения точности прогнозирования настроений сотрудников. Опишите возможную архитектуру ансамбля и объясните, какие модели вы бы включили в ансамбль и почему."\n\t},\n\t{\n\t\t"competence": "Использует программные и технические средства для визуализации больших данных",\n\t\t"question": "Какие методы и инструменты вы бы использовали для визуализации данных и результатов модели ИС ПАУС? Как эти визуализации могут помочь в анализе и принятии решений?"\n\t},\n\t{\n\t\t"competence": "Решает задачи искусственного интеллекта (ИИ)",\n\t\t"question": "Какую роль технологии искусственного интеллекта могут сыграть в системе предиктивной аналитики увольнения сотрудников? Опишите возможные ИИ-решения и их преимущества для данной задачи."\n\t},\n\t{\n\t\t"competence": "Разрабатывает и применяет методы машинного обучения (МО) для решения задач",\n\t\t"question": "Опишите процесс разработки модели машинного обучения для предсказания настроений сотрудников. Какие методы обучения и алгоритмы вы бы использовали и почему?"\n\t},\n\t{\n\t\t"competence": "Применяет нормы этики при использования данных",\n\t\t"question": "Какие этические вопросы могут возникнуть при сборе и использовании данных сотрудников для предсказания их настроений к увольнению? Как вы бы предложили минимизировать эти риски?"\n\t}\n\t]\n}\n```',
            'format_instructions': generation_format_instructions
        },
        {
            'case_desc': 'Вы - ведущий специалист в команде разработки системы поддержки принятия решений в области здравоохранения. Система разрабатывается для крупной сети клиник и предназначена для анализа и прогнозирования исходов лечения пациентов на основе данных электронной медицинской карты, медицинских изображений, лабораторных анализов и генетической информации. Ключевая цель системы - помочь врачам принимать обоснованные решения о назначении терапии, прогнозировании возможных осложнений и оптимизации лечебного процесса. Ваша задача - разработать и внедрить алгоритмы для обработки и анализа больших объемов данных, интерпретировать результаты работы модели и обеспечить контроль качества данных. Помимо этого, важно учитывать этические аспекты, связанные с использованием персональных медицинских данных, и минимизировать риски, связанные с возможными ошибками системы, которые могут повлиять на здоровье пациентов. Также необходимо использовать методы объяснимого ИИ для того, чтобы врачи могли понимать, на чем основаны прогнозы и рекомендации системы, и уверенно принимать решения.',
            'competence_list': ' - Анализирует и интерпретирует результаты работы моделей машинного обучения\n - Разрабатывает и тестирует алгоритмы для обработки больших данных\n - Оценивает и минимизирует риски при использовании искусственного интеллекта\n - Осуществляет контроль качества данных и процессов обучения моделей\n - Использует методы объяснимого ИИ для интерпретации моделей',
            'output': 'Для начала, я проанализирую кейс и указанные компетенции, чтобы сформулировать соответствующие открытые вопросы.\n\n### Анализ компетенций и кейса\n\nКейс описывает разработку системы поддержки принятия решений в здравоохранении, где ключевые задачи включают анализ и прогнозирование исходов лечения на основе различных медицинских данных, включая изображения, лабораторные анализы и генетическую информацию. Важно обеспечить высокое качество данных, интерпретировать результаты модели и минимизировать риски ошибок, поскольку это может повлиять на здоровье пациентов. Также требуется учитывать этические аспекты и использовать методы объяснимого ИИ для повышения доверия врачей к системе.\n\n#### Компетенции:\n1. **Анализирует и интерпретирует результаты работы моделей машинного обучения** - Эта компетенция предполагает умение оценивать выходные данные моделей, определять их значимость и делать выводы на основе этих данных.\n2. **Разрабатывает и тестирует алгоритмы для обработки больших данных** - Компетенция направлена на способность разрабатывать и внедрять алгоритмы, которые могут эффективно работать с большими объемами медицинских данных.\n3. **Оценивает и минимизирует риски при использовании искусственного интеллекта** - Важный аспект, связанный с пониманием потенциальных рисков, которые могут возникнуть при использовании ИИ в здравоохранении, и разработкой стратегий для их минимизации.\n4. **Осуществляет контроль качества данных и процессов обучения моделей** - Подразумевает контроль над качеством данных, используемых для обучения моделей, и мониторинг процесса обучения для обеспечения корректности работы системы.\n5. **Использует методы объяснимого ИИ для интерпретации моделей** - Эта компетенция предполагает использование методов, которые помогают объяснить, как модель пришла к тем или иным выводам, чтобы врачи могли доверять рекомендациям системы.\n\n### Генерация вопросов\n\nНа основе анализа кейса и компетенций, формулирую вопросы, которые проверяют наличие соответствующих компетенций у студентов.\n\n```json\n{\n\"questions\": [\n\t{\n\t\t"competence": "Анализирует и интерпретирует результаты работы моделей машинного обучения",\n\t\t"question": "Представьте, что модель прогнозирования осложнений после хирургического вмешательства показала высокий уровень ложноположительных результатов. Какие шаги вы предпримете для анализа и интерпретации этого результата, чтобы предложить пути улучшения модели?"\n\t},\n\t{\n\t\t"competence": "Разрабатывает и тестирует алгоритмы для обработки больших данных",\n\t\t"question": "Какой подход вы бы выбрали для обработки и анализа данных из медицинских карт пациентов, чтобы эффективно использовать их для прогнозирования исходов лечения? Опишите процесс разработки и тестирования такого алгоритма."\n\t},\n\t{\n\t\t"competence": "Оценивает и минимизирует риски при использовании искусственного интеллекта",\n\t\t"question": "Система, разработанная вами, предполагает принятие решений о назначении терапии. Как вы оцените и минимизируете риски, связанные с возможными ошибками модели, которые могут негативно повлиять на здоровье пациентов?"\n\t},\n\t{\n\t\t"competence": "Осуществляет контроль качества данных и процессов обучения моделей",\n\t\t"question": "При интеграции данных из различных источников для обучения модели были обнаружены значительные различия в форматах и качестве данных. Как вы обеспечите контроль качества данных и гарантируете надежность процесса обучения модели?"\n\t},\n\t{\n\t\t"competence": "Использует методы объяснимого ИИ для интерпретации моделей",\n\t\t"question": "Врачи, использующие вашу систему, хотят понимать, как система пришла к тому или иному прогнозу. Какие методы объяснимого ИИ вы будете использовать, чтобы обеспечить прозрачность и доверие к выводам модели?"\n\t}\n\t]\n}\n```',
            'format_instructions': generation_format_instructions
        },

    ],
}

# === Main pipeline ===

def get_model(primary_model='qwen2-72b', num_examples=3):
    few_shot = True
    if num_examples <= 0:
        few_shot = False
    if primary_model not in model_map:
        raise Exception(f'{primary_model} is not a valid or supported model')
    model_info = model_map[primary_model]
    api_url = model_info['api']
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    
    tokenizer = PreTrainedTokenizerFast.from_pretrained(model_info['tokenizer'], trust_remote_code=True)

    case_prompt_few_shot = ChatPromptTemplate.from_messages(
        [
            ("human", "# Компетенции:\n{competence_list}\n\nТы должен придумать и сгенерировать интерестный кейс для студенческого экзамена по заданным компетенциям, к которому можно задать интерестные и сложные вопросы. Для начала тщательно проанализируй компетенции и используй цепочку размышлений и напиши свой анализ. После этого, сгенерируй кейс по данным компетенциям. Кейс должен содержать все компетенции. Не повторяй уже сгенерированные кейсы.\n\n{format_instructions}"),
            ("ai", "{output}")
        ]
    )
    
    case_few_shot_prompt = FewShotChatMessagePromptTemplate(
        example_prompt=case_prompt_few_shot,
        examples=few_shot_examples['case'][:num_examples],
    )
    
    case_prompt_template = ChatPromptTemplate.from_messages(
        [
            ("system", system_message),
            *([case_few_shot_prompt] if few_shot else []),
            ("human", "# Компетенции:\n{competence_list}\n\nТы должен придумать и сгенерировать интерестный кейс для студенческого экзамена по заданным компетенциям, к которому можно задать интерестные и сложные вопросы. Для начала тщательно проанализируй компетенции и используй цепочку размышлений и напиши свой анализ. После этого, сгенерируй кейс по данным компетенциям. Кейс должен содержать все компетенции. Не повторяй уже сгенерированные кейсы.")
        ]
    )

    case_chain = ChatChain(case_prompt_template, tokenizer, api_url)
    
    def generate_case(competence_list):
        competence_list = '\n'.join([f' - {competence}' for competence in competence_list])

        # Шаг 1: Сгенерировать тему
        case_args = {
            "competence_list": competence_list,
            "format_instructions": get_format_instructions(parser_case)
        }
        print(f"case_args = {case_args}")
        while True:
            try:
                case_result = case_chain.invoke(case_args)
                case_json = str_to_json(case_result)
                case = case_json['case_desc']
                
                print(f"case = {case}")
                return case
            except:
                pass

    def get_generation_prompt():
        generation_prompt_few_shot = ChatPromptTemplate.from_messages(
            [
                ("human", "# Компетенции:\n{competence_list}\n\n# Кейс: {case_desc}\n\nТы должен сгенерировать открытые вопросы для каждой указанной компетенции для данного кейса в рамках студенческого экзамена. Для начала тщательно проанализируй компетенции и кейс и используй цепочку размышлений и напиши свой анализ. После этого, сгенерируй по одному вопросу для каждой компетенции по данному кейсу. Каждый вопрос должен тестировать наличие компетенции у студента, соответвующей вопросу. Не повторяй уже сгенерированные вопросы.\n\n{format_instructions}"),
                ("ai", "{output}")
            ]
        )

        generation_few_shot_prompt = FewShotChatMessagePromptTemplate(
            example_prompt=generation_prompt_few_shot,
            examples=few_shot_examples['generation'][:num_examples],
        )

        generation_prompt_template = ChatPromptTemplate.from_messages(
            [
                ("system", system_message),
                *([generation_few_shot_prompt] if few_shot else []),
                ("human", "# Компетенции:\n{competence_list}\n\n# Кейс: {case_desc}\n\nТы должен сгенерировать открытые вопросы для каждой указанной компетенции для данного кейса в рамках студенческого экзамена. Для начала тщательно проанализируй компетенции и кейс и используй цепочку размышлений и напиши свой анализ. После этого, сгенерируй по одному вопросу для каждой компетенции по данному кейсу. Каждый вопрос должен тестировать наличие компетенции у студента, соответвующей вопросу. Не повторяй уже сгенерированные вопросы."),
            ]
        )

        return ChatChain(generation_prompt_template, tokenizer, api_url)
    
    def validate_questions(generated_questions):
        if not isinstance(generated_questions, list):
            raise Exception('Generated questions is not a list')
        
        for gen_q in generated_questions:
            if 'question' not in gen_q:
                raise Exception('"question" is not in generated question list')
            if 'competence' not in gen_q:
                raise Exception('"competence" is not in generated question list')
    
    
    def generate_questions(case, competence_list):
        competence_list = '\n'.join([f' - {competence}' for competence in competence_list])

        # Шаг 2: Сгенерировать вопрос по кейсу
        generation_chain = get_generation_prompt()
        generation_args = {
            "case_desc": case,
            "competence_list": competence_list,
            "format_instructions": generation_format_instructions
        }
        while True:
            try:
                generated_question_result = generation_chain.invoke(generation_args)
                generated_question_json = str_to_json(generated_question_result)
                generated_questions = generated_question_json['questions']
                print('Generated question:', generated_questions)
                validate_questions(generated_questions)
                break
            except Exception as e:
                print('Failed attempt at question generation:', e)
                print('Gen result', generated_question_result)
                pass
        
        print(f"generated_questions = {generated_questions}")
        
        return generated_questions

    return generate_case, generate_questions
