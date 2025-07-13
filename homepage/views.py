from django.shortcuts import render, redirect
from django_test.settings import STATICFILES_DIRS
from django.http import JsonResponse, HttpResponse
from django.conf import settings
from django.core.paginator import Paginator
import os
import io
import base64
import cv2
import matplotlib.pyplot as plt
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.conf import settings
import requests
import json
from django.http import StreamingHttpResponse
from django.views.decorators.csrf import csrf_exempt
from zhipuai import ZhipuAI
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from urllib.parse import unquote
import math


def homepage(request):
    return render(request, 'homepage.html')


def image_list_page(request):
    image_path = os.path.join(STATICFILES_DIRS[0], 'images')
    image_dir = os.listdir(image_path)
    image_arr = []
    for item in image_dir:
        image_arr.append(os.path.join('/images', item))
    # 分页
    paginator = Paginator(image_arr, 1)  # 每页显示1张图片
    page_number = request.GET.get('page')  # 获取当前页码，默认为第1页
    page_obj = paginator.get_page(page_number)
    return render(request, 'image_list.html', {'page_obj': page_obj})


def video_list_page(request):
    video_path = os.path.join(STATICFILES_DIRS[0], 'videos')
    video_dir = os.listdir(video_path)
    video_arr = [os.path.join('/videos', item) for item in video_dir]
    # 分页
    paginator = Paginator(video_arr, 1)  # 每页显示1个视频
    page_number = request.GET.get('page', 1)  # 获取当前页码，默认为第1页
    page_obj = paginator.get_page(page_number)
    return render(request, 'video_list.html', {'page_obj': page_obj})


def upload(request):
    return render(request, 'upload.html')


def upload_image(request):
    upload_success = False
    image_name = ''
    if (request.method == 'POST'):
        if 'image' not in request.FILES:  # 当用户没有选择文件时的错误处理
            error_message = "请选择要上传的文件。"
            return render(request, 'upload_error.html', {'error_message': error_message})
        image_content = request.FILES['image']
        image_name = image_content.name
        file_path = os.path.join(STATICFILES_DIRS[0], 'images', image_name)
        with open(file_path, 'wb+') as destination:
            for chunk in image_content.chunks():
                destination.write(chunk)
        upload_success = True
    return render(request, 'upload.html', {'image_name': image_name, 'upload_success': upload_success})


def upload_video(request):
    upload_success1 = False
    video_name = ''
    if (request.method == 'POST'):
        if 'video' not in request.FILES:
            error_message = "请选择要上传的文件。"
            return render(request, 'upload_error.html', {'error_message': error_message})
        video_content = request.FILES['video']
        video_name = video_content.name
        file_path = os.path.join(STATICFILES_DIRS[0], 'videos', video_name)
        with open(file_path, 'wb+') as destination:
            for chunk in video_content.chunks():
                destination.write(chunk)
        upload_success1 = True
    return render(request, 'upload.html', {'video_name': video_name, 'upload_success1': upload_success1})


def upload_error(request, upload_error):
    return render(request, 'upload_error.html', {'upload_error': upload_error})


def search_image(request):
    if request.method == 'POST':
        search_content = request.POST.get('search_name')
        image_path = os.path.join(STATICFILES_DIRS[0], 'images')
        image_dir = os.listdir(image_path)
        image_arr = []
        for item in image_dir:
            if item == search_content or search_content == '':
                image_arr.append(os.path.join('/images', item))
        # 分页
        paginator = Paginator(image_arr, 1)  # 每页显示1张图片
        page_number = request.GET.get('page')  # 获取当前页码，默认为第1页
        page_obj = paginator.get_page(page_number)
        return render(request, 'image_list.html', {'search_content': search_content, 'page_obj': page_obj})


def search_video(request):
    if request.method == 'POST':
        search_content = request.POST.get('search_name')
        video_path = os.path.join(STATICFILES_DIRS[0], 'videos')
        video_dir = os.listdir(video_path)
        video_arr = []
        for item in video_dir:
            if item == search_content or search_content == '':
                video_arr.append(os.path.join('/videos', item))
        paginator = Paginator(video_arr, 1)  # 每页显示1张图片
        page_number = request.GET.get('page')  # 获取当前页码，默认为第1页
        page_obj = paginator.get_page(page_number)
        return render(request, 'video_list.html', {'search_content': search_content, 'page_obj': page_obj})


def imageProcessing(request):
    return render(request, 'image_process.html')


def load_images(request):
    image_dir = os.path.join(settings.STATICFILES_DIRS[0], 'images')
    image_files = os.listdir(image_dir)
    image_urls = [os.path.join('static/images', file) for file in image_files]
    return JsonResponse({'image_urls': image_urls})


def grayscale_image(request):  # 灰度化
    url = request.GET.get('key1')
    src_path = url.split('/')[-1]
    image_path = os.path.join(settings.STATICFILES_DIRS[0], src_path)
    des_path = os.path.join(settings.STATICFILES_DIRS[0], 'images_result', src_path)

    if not os.path.exists(des_path):
        color_image = cv2.imread(image_path)
        gray_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
        cv2.imwrite(des_path, gray_image)

    grayscale_image_url = os.path.join('static/images_result', src_path)
    return JsonResponse({'grayscale_image_url': grayscale_image_url})


def equalize_image(request):  # 均衡化
    url = request.GET.get('key1')
    src_path = url.split('/')[-1]
    image_path = os.path.join(settings.STATICFILES_DIRS[0], src_path)
    des_path = os.path.join(settings.STATICFILES_DIRS[0], 'images_result2', src_path)

    if not os.path.exists(des_path):
        color_image = cv2.imread(image_path)
        hsv_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2HSV)
        hsv_image[:, :, 2] = cv2.equalizeHist(hsv_image[:, :, 2])
        equalized_image = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)
        cv2.imwrite(des_path, equalized_image)

    equalized_image_url = os.path.join('static/images_result2', src_path)
    return JsonResponse({'equalized_image_url': equalized_image_url})


def audio_play(request):
    return render(request, 'audio_play.html')


def ai_assistant(request):
    return render(request, 'ai_assistant.html')


@csrf_exempt
def ask_baidu_api(request):
    if request.method == 'POST':
        try:
            api_key = settings.BAIDU_SINGLE_API_KEY
            data = json.loads(request.body)
            user_question = data.get('question')
            if not user_question:
                return JsonResponse({'error': '问题不能为空'}, status=400)

            url = "https://aip.baidubce.com/rpc/2.0/ai_custom/v1/wenxinworkshop/chat/completions"
            headers = {
                'Content-Type': 'application/json',
                'Authorization': f'Bearer {api_key}'
            }

            payload = json.dumps({
                "model": "ernie-3.5-8k",  # 模型
                "messages": [
                    {
                        "role": "user",
                        "content": user_question
                    }
                ],
                "system": "你是一个充满热情、知识渊博的生态和生物多样性科普专家。"
            })

            response = requests.post(url, headers=headers, data=payload, timeout=100)
            response.raise_for_status()

            ai_answer = response.json().get("result")
            return JsonResponse({'answer': ai_answer})

        except Exception as e:
            return JsonResponse({'error': f'服务器内部错误: {e}'}, status=500)

    return JsonResponse({'error': '仅支持POST请求'}, status=405)


# 流式输出，提高观感
# @csrf_exempt
# def ask_streaming_api(request):
#     def event_stream():
#         try:
#             api_key = settings.BAIDU_SINGLE_API_KEY
#             data = json.loads(request.body)
#             user_question = data.get('question')
#             if not user_question:
#                 # 以流的形式发送错误信息
#                 error_data = json.dumps({'error': '问题不能为空'})
#                 yield f"data: {error_data}\n\n"
#                 return
#
#             url = "https://aip.baidubce.com/rpc/2.0/ai_custom/v1/wenxinworkshop/chat/completions"
#             headers = {'Content-Type': 'application/json', 'Authorization': f'Bearer {api_key}'}
#
#             payload = json.dumps({
#                 "model": "ernie-3.5-8k",
#                 "messages": [{"role": "user", "content": user_question}],
#                 "system": "你是一个充满热情、知识渊博的生态和生物多样性科普专家。",
#                 "stream": True  # 流式输出
#             })
#
#             response = requests.post(url, headers=headers, data=payload, stream=True, timeout=100)
#             response.raise_for_status()
#
#             # 处理返回的数据流
#             for chunk in response.iter_content(chunk_size=None):
#                 chunk_str = chunk.decode('utf-8')
#                 # 流式数据通常以 "data: " 开头，直接转发给前端
#                 if chunk_str.startswith("data:"):
#                     yield chunk_str + '\n\n'  # 转发原始的SSE数据块
#
#         except Exception as e:
#             error_data = json.dumps({'error': str(e)})
#             yield f"data: {error_data}\n\n"
#
#     # 返回一个流式响应对象
#     return StreamingHttpResponse(event_stream(), content_type='text/event-stream')


# 利用RAG知识库增强的流式输出
# @csrf_exempt
def ask_streaming_api(request):
    def event_stream():
        try:
            embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
            )
            vector_store = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
            retriever = vector_store.as_retriever()

            data = json.loads(request.body)
            user_question = data.get('question')

            retrieved_docs = retriever.invoke(user_question)
            # 将检索到的文档内容拼接成一个字符串
            context_text = "\n\n".join([doc.page_content for doc in retrieved_docs])
            print(f"已从知识库检索到相关内容。")

            # 手动构建Prompt
            prompt_template = f"""
            你是一个专业的、热情的生态科普助手。请优先根据下面提供的“背景知识”来回答用户的问题。
            如果背景知识里有相关信息，请一定依据它来回答。
            如果背景知识里没有相关内容，就直接利用你的通用知识来回答用户的问题。

            ---
            背景知识:
            {context_text}
            ---

            用户问题:
            {user_question}
            """

            api_key = settings.BAIDU_SINGLE_API_KEY
            url = "https://aip.baidubce.com/rpc/2.0/ai_custom/v1/wenxinworkshop/chat/completions"
            headers = {
                'Content-Type': 'application/json',
                'Authorization': f'Bearer {api_key}'
            }
            payload = json.dumps({
                "model": "ERNIE-Bot-turbo",
                "messages": [{
                    "role": "user",
                    "content": prompt_template
                }],
                "stream": True
            })

            print("准备手动发送流式请求到百度API")
            response = requests.post(url, headers=headers, data=payload, stream=True, timeout=60)
            response.raise_for_status()

            # 手动处理返回的流式数据
            for chunk in response.iter_content(chunk_size=None):
                chunk_str = chunk.decode('utf-8')
                if chunk_str.startswith("data:"):
                    yield chunk_str + '\n\n'

        except Exception as e:
            print(f"程序在TRY块中崩溃！错误是: {e}")
            yield f"data: {json.dumps({'error': str(e)})}\n\n"

    return StreamingHttpResponse(event_stream(), content_type='text/event-stream')


@csrf_exempt
def ai_recognize_image(request):
    if request.method == 'GET':
        try:
            client = ZhipuAI(api_key=settings.ZHIPUAI_API_KEY)

            image_url_from_frontend = request.GET.get('imageUrl')
            image_path_on_server = os.path.join(settings.BASE_DIR, image_url_from_frontend)

            if not os.path.exists(image_path_on_server):
                return JsonResponse({'error': '服务器上找不到文件'}, status=404)

            with open(image_path_on_server, "rb") as f:
                image_base64 = base64.b64encode(f.read()).decode('utf-8')

            prompt_text = "你是一位专业的自然生态学家和科普作家。请详细描述这张图片的内容，如果里面有具体的动植物，请尝试识别出它的种类，并对它进行一段生动有趣的科普介绍。"

            response = client.chat.completions.create(
                model="glm-4v",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": prompt_text
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{image_base64}"
                                }
                            }
                        ]
                    }
                ]
            )
            print(f"完整响应内容: {response} ---")

            ai_answer = response.choices[0].message.content
            return JsonResponse({'description': ai_answer})

        except Exception as e:
            return JsonResponse({'error': str(e)}, status=500)

    return JsonResponse({'error': '仅支持GET请求'}, status=405)


@csrf_exempt
def ai_analyze_video(request):
    if request.method == 'GET':
        try:
            client = ZhipuAI(api_key=settings.ZHIPUAI_API_KEY)
            video_url_from_frontend = request.GET.get('videoUrl')
            decoded_url = unquote(video_url_from_frontend)
            relative_path = decoded_url.lstrip('/')
            video_path_on_server = os.path.join(settings.BASE_DIR, relative_path)

            if not os.path.exists(video_path_on_server):
                return JsonResponse({'error': f'服务器上找不到视频文件: {video_path_on_server}'}, status=404)

            video_capture = cv2.VideoCapture(video_path_on_server)

            total_frames = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = video_capture.get(cv2.CAP_PROP_FPS)

            if fps == 0:
                duration_seconds = 0
            else:
                duration_seconds = total_frames / fps

            BASE_FRAMES = 3
            SECONDS_PER_EXTRA_FRAME = 20
            MAX_FRAMES = 5

            additional_frames = math.floor(duration_seconds / SECONDS_PER_EXTRA_FRAME)
            num_frames_to_extract = min(BASE_FRAMES + additional_frames, MAX_FRAMES)

            if total_frames > 0 and num_frames_to_extract == 0:
                num_frames_to_extract = 1
            num_frames_to_extract = min(num_frames_to_extract, total_frames)

            print(
                f"视频时长: {duration_seconds:.2f}秒, 总帧数: {total_frames}, 计划截取: {num_frames_to_extract}帧")

            frames = []
            if num_frames_to_extract > 0:
                interval = total_frames // num_frames_to_extract
                for i in range(num_frames_to_extract):
                    # 从interval//2开始
                    frame_position = (i * interval) + (interval // 2)

                    video_capture.set(cv2.CAP_PROP_POS_FRAMES, frame_position)
                    success, frame = video_capture.read()
                    if success:
                        _, buffer = cv2.imencode('.jpg', frame)
                        frame_base64 = base64.b64encode(buffer).decode('utf-8')
                        frames.append(frame_base64)

            video_capture.release()

            if not frames:
                return JsonResponse({'error': '无法从视频中截取关键帧'}, status=500)

            prompt_text = "你是一位专业的纪录片分析师。根据下面从一个视频中截取的几张关键画面，请用一段话总结这个视频可能讲述的内容、主题和场景。"

            content_list = [{"type": "text", "text": prompt_text}]
            for frame_b64 in frames:
                content_list.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{frame_b64}"}
                })

            response = client.chat.completions.create(
                model="glm-4v",
                messages=[{"role": "user", "content": content_list}]
            )

            ai_answer = response.choices[0].message.content
            return JsonResponse({'description': ai_answer})

        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"程序在TRY块中崩溃！错误是: {e}")
            return JsonResponse({'error': str(e)}, status=500)

    return JsonResponse({'error': '仅支持GET请求'}, status=405)
