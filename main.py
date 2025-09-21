# bangladesh_nn_keyword_bot.py
import torch
import torch.nn as nn
import torch.nn.functional as F

# Sample Q&A dataset
questions = [
    "capital", "population", "language", "currency",
    "famous place", "national animal", "national bird", "independence day",
    "cumilla"
]

answers = [
    "The capital of Bangladesh is Dhaka.",
    "Bangladesh has a population of approx. 170 million.",
    "The official language is Bengali (Bangla).",
    "The currency used is Bangladeshi Taka (BDT).",
    "Some famous places are: Sundarbans, Cox's Bazar, Srimangal, Paharpur.",
    "The national animal is the Bengal Tiger.",
    "The national bird is the Oriental Magpie Robin.",
    "Bangladesh celebrates Independence Day on 26 March 1971.",
    "Cumilla is a very educated place in Bangladesh."
]

# Simple tokenization (word presence)
vocab = list(set(" ".join(questions).split()))
word_to_idx = {w: i for i, w in enumerate(vocab)}


def question_to_vector(q):
    vec = torch.zeros(len(vocab))
    for w in q.lower().split():
        if w in word_to_idx:
            vec[word_to_idx[w]] = 1
    return vec


# 8-layer feedforward neural network
class BangladeshNNBot(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(BangladeshNNBot, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, hidden_size)
        self.fc5 = nn.Linear(hidden_size, hidden_size)
        self.fc6 = nn.Linear(hidden_size, hidden_size)
        self.fc7 = nn.Linear(hidden_size, hidden_size)
        self.fc8 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = F.relu(self.fc5(x))
        x = F.relu(self.fc6(x))
        x = F.relu(self.fc7(x))
        x = self.fc8(x)  # last layer, raw scores
        return x


# Prepare training data
X_train = torch.stack([question_to_vector(q) for q in questions])
y_train = torch.tensor([i for i in range(len(answers))])

# Initialize model
input_size = len(vocab)
hidden_size = 16
output_size = len(answers)
model = BangladeshNNBot(input_size, hidden_size, output_size)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# Training loop
for epoch in range(1000):
    optimizer.zero_grad()
    outputs = model(X_train)
    loss = criterion(outputs, y_train)
    loss.backward()
    optimizer.step()
    if epoch % 200 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

# Keyword map for random phrasing handling
keyword_answers = {
    "capital": "The capital of Bangladesh is Dhaka.",
    "population": "Bangladesh has a population of approx. 170 million.",
    "language": "The official language is Bengali (Bangla).",
    "currency": "The currency used is Bangladeshi Taka (BDT).",
    "famous": "Some famous places are: Sundarbans, Cox's Bazar, Srimangal, Paharpur.",
    "animal": "The national animal is the Bengal Tiger.",
    "bird": "The national bird is the Oriental Magpie Robin.",
    "independence": "Bangladesh celebrates Independence Day on 26 March 1971.",
    "cumilla": "Cumilla is a very educated place in Bangladesh.",
    "beautiful": "Yes, Bangladesh is beautiful! 🇧🇩",
    "Bagerhat": "Postal Code: 9300, Area: 3914 km², Population: 1380000, Upazilas: 9",
    "Bandarban": "Postal Code: 4600, Area: 4479 km², Population: 388335, Upazilas: 6",
    "Barguna": "Postal Code: 8700, Area: 1682 km², Population: 880000, Upazilas: 7",
    "Barisal": "Postal Code: 8200, Area: 2774 km², Population: 2800000, Upazilas: 10",
    "Bhola": "Postal Code: 8300, Area: 3400 km², Population: 1800000, Upazilas: 8",
    "Bogra": "Postal Code: 5800, Area: 3023 km², Population: 2800000, Upazilas: 12",
    "Brahmanbaria": "Postal Code: 3400, Area: 2465 km², Population: 2500000, Upazilas: 9",
    "Chandpur": "Postal Code: 4200, Area: 1580 km², Population: 2500000, Upazilas: 9",
    "Chattogram": "Postal Code: 4000, Area: 5280 km², Population: 8000000, Upazilas: 15",
    "Chuadanga": "Postal Code: 7100, Area: 1913 km², Population: 1500000, Upazilas: 12",
    "Cumilla": "Postal Code: 3500, Area: 3080 km², Population: 2800000, Upazilas: 17",
    "Dhaka": "Postal Code: 1000, Area: 1463 km², Population: 14734701, Upazilas: 5",
    "Dinajpur": "Postal Code: 5200, Area: 6884 km², Population: 2300000, Upazilas: 13",
    "Faridpur": "Postal Code: 7800, Area: 2370 km², Population: 1500000, Upazilas: 9",
    "Feni": "Postal Code: 3900, Area: 1041 km², Population: 1500000, Upazilas: 6",
    "Gaibandha": "Postal Code: 5700, Area: 3800 km², Population: 1600000, Upazilas: 9",
    "Gazipur": "Postal Code: 1700, Area: 1744 km², Population: 3000000, Upazilas: 4",
    "Gopalganj": "Postal Code: 8100, Area: 1452 km², Population: 1300000, Upazilas: 7",
    "Habiganj": "Postal Code: 3300, Area: 2741 km², Population: 1600000, Upazilas: 9",
    "Jamalpur": "Postal Code: 2000, Area: 1970 km², Population: 2400000, Upazilas: 8",
    "Jashore": "Postal Code: 7400, Area: 2604 km², Population: 2400000, Upazilas: 8",
    "Jhalokathi": "Postal Code: 8500, Area: 1000 km², Population: 700000, Upazilas: 5",
    "Jhenaidah": "Postal Code: 7300, Area: 2210 km², Population: 1500000, Upazilas: 8",
    "Joypurhat": "Postal Code: 5900, Area: 963 km², Population: 600000, Upazilas: 5",
    "Khagrachari": "Postal Code: 4400, Area: 2790 km², Population: 400000, Upazilas: 9",
    "Khulna": "Postal Code: 9000, Area: 4394 km², Population: 1500000, Upazilas: 9",
    "Kishoreganj": "Postal Code: 2300, Area: 2744 km², Population: 3500000, Upazilas: 13",
    "Kurigram": "Postal Code: 5600, Area: 2395 km², Population: 1500000, Upazilas: 9",
    "Kushtia": "Postal Code: 7000, Area: 1625 km², Population: 1200000, Upazilas: 7",
    "Lakshmipur": "Postal Code: 3600, Area: 1550 km², Population: 1500000, Upazilas: 6",
    "Lalmonirhat": "Postal Code: 5500, Area: 1260 km², Population: 1000000, Upazilas: 5",
    "Madaripur": "Postal Code: 7900, Area: 1049 km², Population: 700000, Upazilas: 6",
    "Magura": "Postal Code: 7600, Area: 1028 km², Population: 500000, Upazilas: 5",
    "Manikganj": "Postal Code: 1500, Area: 1340 km², Population: 1000000, Upazilas: 7",
    "Meherpur": "Postal Code: 7100, Area: 802 km², Population: 400000, Upazilas: 3",
    "Moulvibazar": "Postal Code: 3200, Area: 2769 km², Population: 1600000, Upazilas: 9",
    "Munshiganj": "Postal Code: 1500, Area: 982 km², Population: 1000000, Upazilas: 6",
    "Mymensingh": "Postal Code: 2200, Area: 436 km², Population: 410000, Upazilas: 4",
    "Naogaon": "Postal Code: 6500, Area: 3741 km², Population: 3000000, Upazilas: 11",
    "Narail": "Postal Code: 7400, Area: 955 km², Population: 500000, Upazilas: 4",
    "Narayanganj": "Postal Code: 1400, Area: 399 km², Population: 2500000, Upazilas: 5",
    "Narsingdi": "Postal Code: 1600, Area: 1394 km², Population: 2000000, Upazilas: 6",
    "Netrokona": "Postal Code: 2400, Area: 2495 km², Population: 1600000, Upazilas: 10",
    "Nilphamari": "Postal Code: 5300, Area: 1696 km², Population: 1200000, Upazilas: 9",
    "Noakhali": "Postal Code: 4400, Area: 3635 km², Population: 1600000, Upazilas: 9",
    "Pabna": "Postal Code: 6600, Area: 2392 km², Population: 1500000, Upazilas: 10",
    "Panchagarh": "Postal Code: 5100, Area: 1333 km², Population: 1000000, Upazilas: 5",
    "Patuakhali": "Postal Code: 8600, Area: 3583 km², Population: 1500000, Upazilas: 8",
    "Pirojpur": "Postal Code: 8500, Area: 2391 km², Population: 1000000, Upazilas: 6",
    "Rajbari": "Postal Code: 7700, Area: 2382 km², Population: 1100000, Upazilas: 6",
    "Rajshahi": "Postal Code: 6000, Area: 2414 km², Population: 1500000, Upazilas: 9",
    "Rangamati": "Postal Code: 4500, Area: 5500 km², Population: 450000, Upazilas: 6",
    "Rangpur": "Postal Code: 5400, Area: 2384 km², Population: 2500000, Upazilas: 8",
    "Satkhira": "Postal Code: 9400, Area: 3383 km², Population: 2000000, Upazilas: 9",
    "Shariatpur": "Postal Code: 8050, Area: 244 km², Population: 175016, Upazilas: 6",
    "Sherpur": "Postal Code: 5700, Area: 1364 km², Population: 400000, Upazilas: 4",
    "Sirajganj": "Postal Code: 6700, Area: 2491 km², Population: 1600000, Upazilas: 11",
    "Sunamganj": "Postal Code: 3000, Area: 3240 km², Population: 2000000, Upazilas: 13",
    "Sylhet": "Postal Code: 3100, Area: 3465 km², Population: 2500000, Upazilas: 13",
    "Tangail": "Postal Code: 1900, Area: 3444 km², Population: 1600000, Upazilas: 12",
    "Thakurgaon": "Postal Code: 5100, Area: 1722 km², Population: 1000000, Upazilas: 7"


}


# Combined function: NN first, keyword fallback
def get_answer(user_input):
    user_input_lower = user_input.lower()

    # 1️⃣ Keyword detection (all districts + keywords)
    for key, ans in keyword_answers.items():
        if key.lower() in user_input_lower:
            return ans

    # 2️⃣ NN fallback (for original Q&A)
    vec = question_to_vector(user_input).unsqueeze(0)
    with torch.no_grad():
        output = model(vec)
        idx = torch.argmax(output).item()
        return answers[idx]



# Bot interaction
print("\nBD Bot (created by JB) is ready! Type 'exit' to quit.")
while True:
    user_input = input("\nYou: ")
    if user_input.lower() in ["exit", "quit"]:
        print("BD: Bye! Have a nice day 🇧🇩")
        break
    response = get_answer(user_input)
    print(f"BD: {response}")
