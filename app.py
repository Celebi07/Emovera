import cv2
import numpy as np
import pygame
from pygame.locals import QUIT
from keras.models import model_from_json
from keras.preprocessing.image import img_to_array
import time
import random
import pygame.mixer
from pygame.locals import QUIT, KEYDOWN, K_RETURN

# Initialize Pygame
pygame.init()
def draw_random_shapes():
    num_shapes = random.randint(1, 5)
    for _ in range(num_shapes):
        shape = random.choice(["circle", "rectangle"])
        color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        x = random.randint(200, 800)
        y = random.randint(200, 600)
        size = random.randint(30, 100)
        if shape == "circle":
            pygame.draw.circle(window, color, (x, y), size)
        elif shape == "rectangle":
            pygame.draw.rect(window, color, (x, y, size, size))
def display_additional_ideas(font):
    idea_texts = [
        "Emotions are the colors of the soul,",
        "Express them all to be truly whole.",
        "From joy to sorrow, they make us real,",
        "Let your heart guide, let your spirit feel.",
        "Show happiness, a smile so bright,",
        "Or sadness, tears that cleanse the night.",
        "Anger flames, a passionate fire,",
        "Fear and courage, a tightrope wire.",
        "Be surprised, let your spirit dance,",
        "Embrace neutrality, take a chance.",
        "Express disgust, a subtle clue,",
        "Let your emotions shine through."
    ]
    
    x = WINDOW_WIDTH * 0.7
    y = 20
    spacing = 30
    
    display_text("Emotions", x, y, font, GREEN)
    
    for i, idea_text in enumerate(idea_texts):
        display_text(idea_text, x, y + (i + 1) * spacing, font, WHITE)
# Set up Pygame window
WINDOW_WIDTH = 1000
WINDOW_HEIGHT = 700
window = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
pygame.display.set_caption('Emotion Game')

# Emotions and their corresponding indices
emotions = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']

# Colors
WHITE = (255, 255, 255)
GREEN = (0, 255, 0)
RED = (255, 0, 0)

def display_shape_prompt(font):
    display_text("Draw:", 20, 740, font)
    display_text("Shapes", 20, 780, font, GREEN)

def display_color_prompt(font):
    display_text("Use:", 20, 840, font)
    display_text("Colors", 20, 880, font, GREEN)

def display_color_options(font):
    colors = ["Red", "Green", "Blue", "Yellow", "Purple", "Orange"]
    display_text("Color Options:", 20, 940, font)
    y = 980
    for color in colors:
        display_text(color, 20, y, font, (255, 255, 0))
        y += 40

def change_background_color():
    r = random.randint(0, 255)
    g = random.randint(0, 255)
    b = random.randint(0, 255)
    window.fill((r, g, b))

def display_time_remaining(last_prompt_time, prompt_interval, font):
    current_time = time.time()
    time_remaining = int(last_prompt_time + prompt_interval - current_time)
    display_text("Time Remaining:", 20, 300, font)
    display_text(str(max(0, time_remaining)), 20, 340, font, GREEN)

def display_prompt(prompt_emotion, font):
    display_text("Prompt Emotion:", 20, 400, font)
    display_text(prompt_emotion, 20, 440, font, GREEN)

def display_negative_effects(score, font):
    if score < 0:
        display_text("Watch out!", 20, 500, font, RED)
        display_text("Negative Score", 20, 540, font, RED)

def display_positive_effects(score, font):
    if score > 0:
        display_text("Keep it up!", 20, 600, font, GREEN)
        display_text("Positive Score", 20, 640, font, GREEN)

def display_instruction(font):
    instruction_text = "Make the face as prompted to earn points!"
    display_text(instruction_text, 20, 680, font, WHITE)

def draw_random_shapes():
    num_shapes = random.randint(1, 5)
    for _ in range(num_shapes):
        shape = random.choice(["circle", "rectangle"])
        color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        x = random.randint(200, 800)
        y = random.randint(200, 600)
        size = random.randint(30, 100)
        if shape == "circle":
            pygame.draw.circle(window, color, (x, y), size)
        elif shape == "rectangle":
            pygame.draw.rect(window, color, (x, y, size, size))

def display_shape_prompt(font):
    display_text("Draw:", 20, 740, font)
    display_text("Shapes", 20, 780, font, GREEN)

def display_color_prompt(font):
    display_text("Use:", 20, 840, font)
    display_text("Colors", 20, 880, font, GREEN)

def display_color_options(font):
    colors = ["Red", "Green", "Blue", "Yellow", "Purple", "Orange"]
    display_text("Color Options:", 20, 940, font)
    y = 980
    for color in colors:
        display_text(color, 20, y, font, (255, 255, 0))
        y += 40

def change_background_color():
    r = random.randint(0, 255)
    g = random.randint(0, 255)
    b = random.randint(0, 255)
    window.fill((r, g, b))

def setup_pygame_font(size):
    return pygame.font.Font(None, size)

def display_text(text, x, y, font, color=WHITE):
    text_surface = font.render(text, True, color)
    window.blit(text_surface, (x, y))

def display_emotion_text(emotion, font):
    display_text("Perform:", 20, 100, font)
    display_text(emotion, 20, 140, font, GREEN)

def display_score(score, font):
    display_text("Score:", 20, 200, font)
    score_color = GREEN if score >= 0 else RED
    display_text(str(score), 20, 240, font, score_color)

def draw_detected_emotion(test_image, x, y, h, detected_emotion):
    label = f"Detected: {detected_emotion}"
    cv2.putText(test_image, label, (int(x), int(y) + h + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, GREEN, 2)

def main():
    # Inside the main loop
    for event in pygame.event.get():
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_SPACE:  # Change background color on SPACE key press
                change_background_color()

    pygame.mixer.init()
    pygame.mixer.music.load("music.mp3")
    model = model_from_json(open("model.json", "r").read())
    pygame.mixer.music.play(-1)
    model.load_weights('best_model.h5')
    face_haar_cascade = cv2.CascadeClassifier('./haarcascade_frontalface_default.xml')
    cap = cv2.VideoCapture(0)
    
    last_prompt_time = time.time()
    prompt_interval = 30  # Prompt for emotion every 30 seconds
    prompt_emotion = random.choice(emotions)
    score = 0

    font_small = setup_pygame_font(22)
    font_medium = setup_pygame_font(32)
    font_large = setup_pygame_font(42)

    # while True:
    #     for event in pygame.event.get():
    #         if event.type == pygame.KEYDOWN:
    #             if event.key == pygame.K_SPACE:
    #                 change_background_color()
                    
    #     valid, test_image = cap.read()
    #     if not valid:
    #         break

    while True:
        for event in pygame.event.get():
            if event.type == QUIT:
                cap.release()
                cv2.destroyAllWindows()
                pygame.quit()
                print("Total Score:", score)
                return

        valid, test_image = cap.read()
        if not valid:
            break

        test_image = cv2.rotate(test_image, cv2.ROTATE_90_COUNTERCLOCKWISE)
        # Clear the right side of the screen
        pygame.draw.rect(window, (0, 0, 0), (WINDOW_WIDTH * 0.7, 0, WINDOW_WIDTH * 0.3, WINDOW_HEIGHT))

        # Display additional ideas
        display_additional_ideas(font_small)
        gray_image = cv2.cvtColor(test_image, cv2.COLOR_BGR2GRAY)

        faces_detected = face_haar_cascade.detectMultiScale(gray_image)
        for (x, y, w, h) in faces_detected:
            cv2.rectangle(test_image, (x, y), (x + w, y + h), (255, 0, 0))
            roi_gray = gray_image[y:y + h, x:x + w]
            roi_gray = cv2.resize(roi_gray, (48, 48))
            image_pixels = img_to_array(roi_gray)
            image_pixels = np.expand_dims(image_pixels, axis=0)
            predictions = model.predict(image_pixels)
            max_index = np.argmax(predictions[0])

            detected_emotion = emotions[max_index]
            score_change = evaluate_emotion(prompt_emotion, detected_emotion)
            score += score_change

            draw_detected_emotion(test_image, x, y, h, detected_emotion)

        resize_image = cv2.resize(test_image, (WINDOW_WIDTH, WINDOW_HEIGHT))
        pygame_image = cv2.cvtColor(resize_image, cv2.COLOR_BGR2RGB)
        pygame_surface = pygame.surfarray.make_surface(pygame_image)
        window.blit(pygame_surface, (0, 0))

        display_emotion_text(prompt_emotion, font_large)
        display_score(score, font_medium)
        display_time_remaining(last_prompt_time, prompt_interval, font_medium)
        # display_prompt(prompt_emotion, font_medium)
        display_negative_effects(score, font_medium)
        display_positive_effects(score, font_medium)
        display_instruction(font_small)
        display_shape_prompt(font_medium)
        display_color_prompt(font_medium)
        display_color_options(font_small)
        pygame.display.flip()

        # Check if it's time for a new prompt
        current_time = time.time()
        if current_time - last_prompt_time >= prompt_interval:
            last_prompt_time = current_time
            prompt_emotion = random.choice(emotions)
            # print(f"Perform: {prompt_emotion}")
        
        # Apply additional functionalities based on the prompt
            if prompt_emotion == "surprise":
                draw_random_shapes()
            elif prompt_emotion == "happy":
                display_shape_prompt(font_medium)
            elif prompt_emotion == "neutral":
                display_color_prompt(font_medium)
                display_color_options(font_small)
    if score > 50:
            pygame.quit()
            print("Congratulations! You scored more than 50 points. Game over.")
            return
def evaluate_emotion(prompt_emotion, detected_emotion):
    if detected_emotion == prompt_emotion:
        print("+1 Point")
        return 1
    else:
        print("-1 Point")
        return -1
    
    
if __name__ == '__main__':
    main()
