import os
import shutil
import requests
from zipfile import ZipFile

FILE_ID = '1D1_lUIucMWqSDqZokohl8Aqd2G5K-LIz'
TARGET_PATH = 'data/mask-fpt-ai.zip'


def download_file_from_google_drive(id, destination):
    URL = "https://docs.google.com/uc?export=download"

    session = requests.Session()

    response = session.get(URL, params={'id': id}, stream=True)
    token = get_confirm_token(response)

    if token:
        params = {'id': id, 'confirm': token}
        response = session.get(URL, params=params, stream=True)

    save_response_content(response, destination)


def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value

    return None


def save_response_content(response, target_path):
    CHUNK_SIZE = 32768

    with open(target_path, "wb") as f:
        TOTAL_LENGTH = int(response.headers.get('content-length'))
        print("Total size: ", TOTAL_LENGTH)
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk:  # filter out keep-alive new chunks
                f.write(chunk)


if __name__ == '__main__':
    print('Start download data')
    if os.path.exists('./data'):
        shutil.rmtree('./data')
    os.makedirs('./data')
    download_file_from_google_drive(FILE_ID, TARGET_PATH)
    print('Download done')

    with ZipFile('./data/mask-fpt-ai.zip', 'r') as zip_file:
        zip_file.extractall('./data')
    print('Success unzip file')
