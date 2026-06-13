import json
import asyncio
from exact.config import get_settings
from exact.llm_client import build_json_client_from_settings
from exact.type1.refiner import Type1Refiner
from exact.type1.refiner.refiner import _format_items, _is_safe_premise_rewrite
from exact.type1.prompts import get_system_prompt_refiner

async def main():
    settings = get_settings()
    # Force use of remote client settings if available, or print config
    print("LLM URL:", settings.llm_base_url)
    print("LLM Model:", settings.llm_model)
    
    client = build_json_client_from_settings()
    if client is None:
        print("Error: VLLMJsonClient could not be built from settings.")
        return
        
    items = [
        {"id": "premise-1", "nl": "Students who have completed the core curriculum and passed the science assessment are qualified for advanced courses.", "fol": "∀x.(Complete(x, CoreCurriculum) IMPLIES (Pass(x, ScienceAssessment) IMPLIES Qualifies(x, AdvancedCourses)))"},
        {"id": "premise-2", "nl": "Students who are qualified for advanced courses and have completed research methodology are eligible for the international program.", "fol": "∀x.(Qualifies(x, AdvancedCourses) IMPLIES (Complete(x, ResearchMethodology) IMPLIES Eligible(x, InternationalProgram)))"},
        {"id": "premise-3", "nl": "Students who have passed the language proficiency exam are eligible for the international program.", "fol": "∀x.(Pass(x, LanguageProficiencyExam) IMPLIES Eligible(x, InternationalProgram))"},
        {"id": "premise-4", "nl": "Students who are eligible for the international program and have completed a capstone project are awarded an honors diploma.", "fol": "∀x.(Eligible(x, InternationalProgram) IMPLIES (Complete(x, CapstoneProject) IMPLIES Awarded(x, HonorDiploma)))"},
        {"id": "premise-5", "nl": "Students who have been awarded an honors diploma and have completed community service qualify for the university scholarship.", "fol": "∀x.(Awarded(x, HonorDiploma) IMPLIES (Complete(x, CommunityService) IMPLIES Qualifies(x, UniversityScholarship)))"},
        {"id": "premise-6", "nl": "Students who have been awarded an honors diploma and have received a faculty recommendation qualify for the university scholarship.", "fol": "∀x.(Awarded(x, HonorDiploma) IMPLIES (HasReceived(x, FacultyRecommendation) IMPLIES Qualifies(x, UniversityScholarship)))"},
        {"id": "premise-7", "nl": "Sophia has completed the core curriculum.", "fol": "Complete(Sophia, CoreCurriculum)"},
        {"id": "premise-8", "nl": "Sophia has passed the science assessment.", "fol": "Pass(Sophia, ScienceAssessment)"},
        {"id": "premise-9", "nl": "Sophia has completed the research methodology course.", "fol": "Complete(Sophia, ResearchMethodologyCourse)"},
        {"id": "premise-10", "nl": "Sophia has completed her capstone project.", "fol": "Complete(Sophia, CapstoneProject)"},
        {"id": "premise-11", "nl": "Sophia has completed the required community service hours.", "fol": "Complete(Sophia, RequiredCommunityServiceHours)"},
        {"id": "C", "nl": "Sophia is eligible for the international program", "fol": "Eligible(Sophia, InternationalProgram)"}
    ]
    
    user_content = _format_items(items)
    print("Formatting user prompt...")
    
    schema = {
        "type": "object",
        "properties": {
            "corrections": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "id": {"type": "string"},
                        "rephrased": {"type": "string"},
                    },
                    "required": ["id", "rephrased"],
                },
            }
        },
        "required": ["corrections"],
    }
    
    try:
        raw_res = await client.complete_json(
            messages=[
                {"role": "system", "content": get_system_prompt_refiner()},
                {"role": "user", "content": user_content},
            ],
            temperature=0.0,
            max_tokens=1024,
            json_schema=schema,
        )
        print("Raw Refiner Response:")
        print(json.dumps(raw_res, indent=2))
        
        print("\nFiltering through _is_safe_premise_rewrite:")
        originals = {item["id"]: item["nl"] for item in items}
        for c in raw_res.get("corrections", []):
            cid = c.get("id")
            rephrased = c.get("rephrased")
            original = originals.get(cid, "")
            is_safe = _is_safe_premise_rewrite(cid, original, rephrased)
            print(f"- ID: {cid} | Rephrased: {rephrased} | Original: {original} | Safe: {is_safe}")
            
    except Exception as e:
        print("Exception occurred during call:", e)

if __name__ == "__main__":
    asyncio.run(main())
