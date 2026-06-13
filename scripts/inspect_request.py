import json
import httpx
import asyncio

Z3_URL = "https://api.iamphuckhang.dev/z3"

async def main():
    payload = {
        "id": "logic_0001_00",
        "premises": [
            "Students who have completed the core curriculum and passed the science assessment are qualified for advanced courses.",
            "Students who are qualified for advanced courses and have completed research methodology are eligible for the international program.",
            "Students who have passed the language proficiency exam are eligible for the international program.",
            "Students who are eligible for the international program and have completed a capstone project are awarded an honors diploma.",
            "Students who have been awarded an honors diploma and have completed community service qualify for the university scholarship.",
            "Students who have been awarded an honors diploma and have received a faculty recommendation qualify for the university scholarship.",
            "Sophia has completed the core curriculum.",
            "Sophia has passed the science assessment.",
            "Sophia has completed the research methodology course.",
            "Sophia has completed her capstone project.",
            "Sophia has completed the required community service hours."
        ],
        "query": "Based on the above premises, which is the strongest conclusion?",
        "options": {
            "A": "Sophia qualifies for the university scholarship",
            "B": "Sophia needs a faculty recommendation to qualify for the scholarship",
            "C": "Sophia is eligible for the international program",
            "D": "Sophia needs to pass the language proficiency exam to get an honors diploma"
        }
    }
    
    async with httpx.AsyncClient() as client:
        r = await client.post(Z3_URL, json=payload, timeout=60.0)
        print("Status Code:", r.status_code)
        if r.status_code == 200:
            print(json.dumps(r.json(), indent=2))
        else:
            print("Response:", r.text)

if __name__ == "__main__":
    asyncio.run(main())
