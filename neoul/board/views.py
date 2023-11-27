from django.core.files.storage import FileSystemStorage
from django.shortcuts import render, redirect
from django.db.models import Q
from .forms import PostForm, SearchForm
from .models import Post
import base64
from multiprocessing import Process
from django.core.paginator import Paginator

from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, WebDriverException


from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.agents.agent_toolkits import create_retriever_tool
from langchain.agents.agent_toolkits import create_conversational_retrieval_agent
from langchain.chat_models import ChatOpenAI
from langchain.agents.openai_functions_agent.agent_token_buffer_memory import (
    AgentTokenBufferMemory,
)
from langchain.agents.openai_functions_agent.base import OpenAIFunctionsAgent
from langchain.prompts import MessagesPlaceholder
from langchain.schema.messages import SystemMessage
from langchain.agents import AgentExecutor
from langchain.docstore.document import Document
from langchain.document_loaders.base import BaseLoader

from time import sleep
from typing import List, Optional
import time
import requests
import nest_asyncio
import re
import logging

logger = logging.getLogger(__name__)

nest_asyncio.apply()

limited_cdata_texts = None

final_response = None
law_name = None
Answer = None


class wait_for_text_change:
    def __init__(self, locator, expected_text):
        self.locator = locator
        self.expected_text = expected_text

    def __call__(self, driver):
        element_text = driver.find_element(*self.locator).text
        return element_text != self.expected_text


class TextLoader(BaseLoader):
    """Load text data directly.

    Args:
        text_data: String containing the text data.
        source: Optional source information for the text data.
    """

    def __init__(
            self,
            text_data: str,
            source: Optional[str] = None
    ):
        """Initialize with text data."""
        self.text_data = text_data
        self.source = source

    def load(self) -> List[Document]:
        """Load from text data."""
        try:
            text = self.text_data
        except Exception as e:
            raise RuntimeError("Error processing text data") from e

        metadata = {"source": self.source}
        return [Document(page_content=text, metadata=metadata)]


# 특수 키워드 변환 규칙

conversion_rules = {
    '사생활': '형법',
    '녹음': '형법',
    '명의 도용': '전자통신사업법',
    '명의': '형법',
    '도용': '형법',
    '임대차': '주택임대차보호법',
    '사기': '형법',
    '모욕': '형법'
    # 다른 규칙들도 추가 가능
}

# 법령 Text 파일 저장 경로
# TextFilePath = '/home/user/exercise_j/AIchatbot-Neoul/law_example_2.txt'

# 질문 입력값
input_data = ""
class TextLoader(BaseLoader):
    """Load text data directly.

    Args:
        text_data: String containing the text data.
        source: Optional source information for the text data.
    """

    def __init__(
            self,
            text_data: str,
            source: Optional[str] = None
    ):
        """Initialize with text data."""
        self.text_data = text_data
        self.source = source

    def load(self) -> List[Document]:
        """Load from text data."""
        try:
            text = self.text_data
        except Exception as e:
            raise RuntimeError("Error processing text data") from e

        metadata = {"source": self.source}
        return [Document(page_content=text, metadata=metadata)]


# 특수 키워드 변환 규칙

conversion_rules = {
    '사생활': '형법',
    '녹음': '형법',
    '명의 도용': '전자통신사업법',
    '명의': '형법',
    '도용': '형법',
    '임대차': '주택임대차보호법',
    '사기': '형법'
    # 다른 규칙들도 추가 가능
}

# 법령 Text 파일 저장 경로
# TextFilePath = '/home/user/exercise_j/AIchatbot-Neoul/law_example_2.txt'

# 질문 입력값
input_data = ""


def generate_law_keyword(input_data, conversion_rules):
    global final_response

    def handle_sensitive_response(response):
        # 모델의 응답이 공백인 경우 민감한 내용으로 간주
        if response.strip().endswith("the most important keyword for a law database would be:"):
            return "죄송하지만, 이 주제에 대해서는 법률적 조언을 제공할 수 없습니다. 전문가의 도움을 받으시길 권장합니다."
        else:
            return response

    # 프롬프트 템플릿 설정
    prompt = ChatPromptTemplate.from_template(
        "Given the input, extract most important one keyword for the law database only in Korean and must be 2 characters long, not a single character. Input: {input}\nKeywords:")



    # 체인 설정: 모델 출력을 키워드로 제한
    chain = prompt | model.bind(stop=["\n"])

    # 체인 실행
    result = chain.invoke({"input": input_data})
    final_response = handle_sensitive_response(result.content)

    for keyword, new_value in conversion_rules.items():
        if keyword in final_response:
            final_response = new_value

    for keyword, new_value in conversion_rules.items():
        if '법률' == final_response or '법조문' == final_response or '법령' == final_response:
            final_response = '형법'

    return final_response


def process_law_info(final_response):
    global limited_cdata_texts
    global law_name

    def extract_cdata(xml_data):
        cdata_sections = re.findall(r'<!\[CDATA\[(.*?)\]\]>', xml_data, re.DOTALL)
        return [cdata.strip() for cdata in cdata_sections]

    def limit_tokens(texts, max_tokens=12000):
        tokenized_texts = [word for text in texts for word in text.split()]
        return ' '.join(tokenized_texts[:max_tokens])

    def search_law(response, retries=5):
        for attempt in range(retries):
            try:
                options = webdriver.ChromeOptions()
                options.add_argument('--headless')
                options.add_argument('--no-sandbox')
                options.add_argument('--disable-dev-shm-usage')
                if 'driver' in globals():
                    driver.quit()

                driver = webdriver.Chrome(options=options)
                print(f"검색 시도: {attempt + 1}, 검색어: '{response}'")
                driver.get("https://glaw.scourt.go.kr/wsjo/lawod/sjo120.do")

                if 'original_text' in globals():
                    original_text = None

                original_text = driver.find_element(By.CSS_SELECTOR, 'h3.search_result_num').text

                search_box = WebDriverWait(driver, 10).until(
                    EC.presence_of_element_located((By.NAME, "srchw"))
                )
                search_box.clear()
                driver.execute_script("arguments[0].value = arguments[1];", search_box, response)
                search_box.send_keys(Keys.RETURN)

                WebDriverWait(driver, 10).until(
                    # wait_for_text_change 함수는 정의되어 있어야 합니다.
                    wait_for_text_change((By.CSS_SELECTOR, 'h3.search_result_num'), original_text)
                )

                popularity_button = WebDriverWait(driver, 10).until(
                    EC.element_to_be_clickable((By.CSS_SELECTOR, "a.btn_type_5[name='sort_popularity']"))
                )
                popularity_button.click()

                time.sleep(5)

                first_result = WebDriverWait(driver, 10).until(
                    EC.presence_of_element_located((By.CSS_SELECTOR, 'td a[name="listCont"] strong'))
                )
                print(f"검색 완료, 첫 번째 결과: {first_result.text}")
                law_name = first_result.text
                return law_name
            except (TimeoutException, WebDriverException) as e:
                print(f"재시도 {attempt + 1}/{retries}, 오류: {e}")
                continue
        print("검색 실패, 결과를 찾을 수 없음.")
        return None

    def fetch_data(url, params, max_retries=10, delay=1):

        """지정된 횟수만큼 요청을 재시도하는 함수"""
        for attempt in range(max_retries):
            try:
                response = requests.get(url, params=params)
                response.raise_for_status()  # 상태 코드가 200이 아닌 경우 예외를 발생시킵니다.
                return response.text
            except requests.RequestException as e:
                print(f"요청 실패 (시도 {attempt + 1}/{max_retries}): {e}")
                sleep(delay)  # 지정된 시간만큼 대기 후 다시 시도
        return None  # 모든 시도가 실패한 경우 None을 반환

    law_name = search_law(final_response)
    if law_name:
        print(f'국가법령정보센터에서 {law_name}에 관한 정보를 불러옵니다.')

        # API의 기본 URL 설정
        base_url = "http://www.law.go.kr/DRF/lawService.do"

        # 요청에 필요한 파라미터 설정
        params = {
            'OC': 'cwindy200',  # 사용자 ID
            'target': 'law',  # 서비스 대상
            'LM': law_name,  # 법령 마스터 번호
            'type': 'XML'  # 출력 형태 (HTML 또는 XML)
        }

        # 함수를 사용하여 데이터 가져오기
        response_text = fetch_data(base_url, params)
        if response_text:
            cdata_texts = extract_cdata(response_text)

            # 토큰 제한 적용
            limited_cdata_texts = limit_tokens(cdata_texts)

            print(f"법령 텍스트 변환: {limited_cdata_texts}")
            return limited_cdata_texts
        else:
            print("모든 요청이 실패했습니다.")
            return None
    else:
        print("정보를 찾을 수 없습니다.")
        return None

def execute_legal_advice_agent(input_data):
    global Answer
    if 'db' in globals():
        db.delete()
    loader = TextLoader(limited_cdata_texts)
    documents = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=50000, chunk_overlap=0)
    texts = text_splitter.split_documents(documents)
    embeddings = OpenAIEmbeddings()
    db = FAISS.from_documents(texts, embeddings)
    retriever = db.as_retriever()
    tool = create_retriever_tool(
        retriever,
        "search_legal_advice",
        "searches and returns answers regarding legal advice and information from Document",
    )
    tools = [tool]

    llm = ChatOpenAI(temperature=0, model="gpt-4-1106-preview", )

    memory_key = "history"

    memory = AgentTokenBufferMemory(memory_key=memory_key, llm=llm)

    system_message = SystemMessage(
        content=(
            "Must not repeat the content found in the Document verbatim."
            "Must use tools to look up relevant information from the Document."
            "Must double-check to ensure the grammar of the Korean answer is correct."
        )
    )

    prompt = OpenAIFunctionsAgent.create_prompt(
        system_message=system_message,
        extra_prompt_messages=[MessagesPlaceholder(variable_name=memory_key)],
    )

    agent = OpenAIFunctionsAgent(llm=llm, tools=tools, prompt=prompt)

    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        memory=memory,
        verbose=True,
        return_intermediate_steps=True,
    )

    result = agent_executor({"input": input_data})

    for message in result['history']:
        if message.__class__.__name__ == 'AIMessage':
            Answer = message.content
            print(message.content)
            return Answer

def integrated_law_process(input_data):
    # 전역 변수 'conversion_rules' 사용
    global conversion_rules

    # Step 1: 법률 관련 키워드 생성
    final_response = generate_law_keyword(input_data, conversion_rules)

    # Step 2: 법률 정보 처리 및 텍스트 파일 생성
    process_law_info(final_response)

    # Step 3: 법률 자문 에이전트 실행
    execute_legal_advice_agent(input_data)





from neoul import settings
import os.path
import urllib
from django.http import HttpResponse
import mimetypes


def file_download(request, file_name):
    path = request.GET.get('path')
    file_path = os.path.join(settings.MEDIA_ROOT, path)
    file_name = urllib.parse.quote(file_name.encode('utf-8'))
    tmp = file_path.split("\\media\\")
    message_bytes = file_name.encode('utf-8')
    base64_bytes = base64.b64encode(message_bytes)
    base64_message = base64_bytes.decode('utf-8')
    if os.path.exists(file_path):
        with open(file_path, 'rb') as fh:
            response = HttpResponse(fh.read(), content_type=mimetypes.guess_type(file_path)[0])
            response['Content-Disposition'] = 'attachment; filename=%s' % file_name
            print(response['Content-Disposition'])
            return response

    else:
        message = '알 수 없는 오류가 발생했습니다.'
        return HttpResponse("<script>alert('" + message + "');history.back()'</script>")


def create(request):
    global contents, keyword1, keyword2
    if request.method == "GET":
        postForm = PostForm()
        context = {'postForm': postForm}
        return render(request, "board/create.html", context)

    elif request.method == "POST":
        postForm = PostForm(request.POST)
        if postForm.is_valid():
            post = postForm.save(commit=False)
            integrated_law_process(post.title)
            print(final_response)
            print(law_name)
            print(Answer)
            print(limited_cdata_texts)
            post.contents = Answer
            post.keyword1 = final_response
            post.keyword2 = law_name
            post.save()

            return redirect('/read/' + str(post.id))


        else:
            return redirect('/create')


def recommend(request, bid):
    post = Post.objects.get(id=bid)
    post.recommend += 1
    post.save()
    context = {'post': post}
    return render(request, 'board/read.html', context)


def search(request, cid, name):
    if request.method == "GET":
        searchForm = SearchForm()
        if cid == 0:
            posts = Post.objects.filter(Q(keyword1__contains=name) | Q(keyword2__contains=name)
                                        | Q(title__contains=name) | Q(contents__contains=name)).order_by('-id')
        if cid == 1:
            posts = Post.objects.filter(Q(keyword1__contains=name) | Q(keyword2__contains=name)).order_by('-id')
        elif cid == 2:
            posts = Post.objects.filter(title__contains=name).order_by('-id')
        elif cid == 3:
            posts = Post.objects.filter(contents__contains=name).order_by('-id')

        page = request.GET.get('page', '1')  # GET 방식으로 정보를 받아오는 데이터
        paginator = Paginator(posts, '10')  # Paginator(분할될 객체, 페이지 당 담길 객체수)
        page_obj = paginator.page(page)  # 페이지 번호를 받아 해당 페이지를 리턴 get_page 권장
        index = page_obj.number - 1
        max_index = len(paginator.page_range)
        start_index = index - 2 if index >= 2 else 0
        if index < 2:
            end_index = 5 - start_index
        else:
            end_index = index + 3 if index <= max_index - 3 else max_index
        page_range = list(paginator.page_range[start_index:end_index])
        context = {'posts': page_obj, 'result_list': page_obj, 'numbers': range(1, 10),
                   'page_range': page_range, 'max_index': max_index - 2, 'searchForm': searchForm, 'page': 'search',
                   'name': name, 'cid': cid}

        return render(request, 'board/list.html', context)



    elif request.method == "POST":

        searchForm = SearchForm(request.POST)

        if searchForm.is_valid():

            search = searchForm.save(commit=False)

            search.save()

            return redirect('/search/' + str(search.type) + '/' + search.search)


        else:

            return redirect(request.META['HTTP_REFERER'])


def list1(request):
    if request.method == "GET":
        searchForm = SearchForm()
        posts = Post.objects.filter().order_by('-id')
        page = request.GET.get('page', '1')  # GET 방식으로 정보를 받아오는 데이터
        paginator = Paginator(posts, '4')  # Paginator(분할될 객체, 페이지 당 담길 객체수)
        page_obj = paginator.page(page)  # 페이지 번호를 받아 해당 페이지를 리턴 get_page 권장
        postImp = Post.objects.filter().order_by('-id')

        index = page_obj.number - 1
        max_index = len(paginator.page_range)
        start_index = index - 2 if index >= 2 else 0
        if index < 2:
            end_index = 5 - start_index
        else:
            end_index = index + 3 if index <= max_index - 3 else max_index
        page_range = list(paginator.page_range[start_index:end_index])

        context = {'posts': page_obj, 'result_list': page_obj, 'postImp': postImp,
                   'numbers': range(1, 10), 'page_range': page_range, 'max_index': max_index - 2,
                   'searchForm': searchForm, 'page': 'home', 'cid': "0", }
        return render(request, 'board/list.html', context)
    elif request.method == "POST":
        searchForm = SearchForm(request.POST)
        if searchForm.is_valid():

            search = searchForm.save(commit=False)
            search.save()
            return redirect('/search/' + str(search.type) + '/' + search.search)

        else:
            print(searchForm.errors)
            return redirect(request.META['HTTP_REFERER'])


def list2(request):
    if request.method == "GET":
        searchForm = SearchForm()
        posts = Post.objects.filter().order_by('-id')
        page = request.GET.get('page', '1')  # GET 방식으로 정보를 받아오는 데이터
        paginator = Paginator(posts, '10')  # Paginator(분할될 객체, 페이지 당 담길 객체수)
        page_obj = paginator.page(page)  # 페이지 번호를 받아 해당 페이지를 리턴 get_page 권장
        postImp = Post.objects.filter().order_by('-id')

        index = page_obj.number - 1
        max_index = len(paginator.page_range)
        start_index = index - 2 if index >= 2 else 0
        if index < 2:
            end_index = 5 - start_index
        else:
            end_index = index + 3 if index <= max_index - 3 else max_index
        page_range = list(paginator.page_range[start_index:end_index])

        context = {'posts': page_obj, 'result_list': page_obj, 'postImp': postImp,
                   'numbers': range(1, 10), 'page_range': page_range, 'max_index': max_index - 2,
                   'searchForm': searchForm, 'page': 'list'}
        return render(request, 'board/list.html', context)
    elif request.method == "POST":
        searchForm = SearchForm(request.POST)
        if searchForm.is_valid():

            search = searchForm.save(commit=False)
            search.save()
            return redirect('/search/' + str(search.type) + '/' + search.search)

        else:
            print(searchForm.errors)
            return redirect(request.META['HTTP_REFERER'])


def read(request, bid):
    post = Post.objects.get(id=bid)
    post.count += 1
    post.save()
    context = {'post': post}
    return render(request, 'board/read.html', context)


def update(request, bid):
    post = Post.objects.get(id=bid)
    if request.method == "GET":
        postForm = PostForm(instance=post)
        context = {'postForm': postForm}
        return render(request, "board/create.html", context)
    elif request.method == "POST":
        postForm = PostForm(request.POST, instance=post)
        if postForm.is_valid():
            post = postForm.save(commit=False)
            post.save()

        return redirect('/list')

    return redirect('/list')


def terms(request):
    return render(request, 'board/terms.html')
