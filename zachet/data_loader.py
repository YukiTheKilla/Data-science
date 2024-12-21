import os
import pandas as pd
import aiohttp
import asyncio

CSV_FILE = '/Users/yuki/Documents/GitHub/Data science/zachet/messor labels.csv'
OUTPUT_FOLDER = '/Users/yuki/Documents/GitHub/datasets/messor'

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

df = pd.read_csv(CSV_FILE, sep=';')


async def download_image(session, image_id, image_url, species_folder):
    """Асинхронная функция для загрузки изображения."""
    image_path = os.path.join(species_folder, f"{image_id}.jpg")
    
    try:
        async with session.get(image_url) as response:
            if response.status == 200:
                with open(image_path, 'wb') as file:
                    while chunk := await response.content.read(1024):
                        file.write(chunk)
                print(f"Изображение {image_id} сохранено")
            else:
                print(f"Ошибка при скачивании {image_url}: статус {response.status}")
    except Exception as e:
        print(f"Ошибка: {e}")


async def main():
    """Главная асинхронная функция."""
    async with aiohttp.ClientSession() as session:
        tasks = []
        for index, row in df.iterrows():
            image_id = row['id']
            image_url = row['image_url']
            scientific_name = row['scientific_name'].replace(' ', '_')
            
            species_folder = os.path.join(OUTPUT_FOLDER, scientific_name)
            os.makedirs(species_folder, exist_ok=True)
            
            task = download_image(session, image_id, image_url, species_folder)
            tasks.append(task)
        
        # Запуск всех задач параллельно
        await asyncio.gather(*tasks)


if __name__ == '__main__':
    asyncio.run(main())
    print("Все изображения обработаны.")