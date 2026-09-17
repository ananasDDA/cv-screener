"""Persona grid: hand-picked specs that guarantee diversity across role, level, geography,
stack, education and languages. The LLM fills in the narrative; identity comes from Faker
seeded per persona so regeneration is deterministic."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from faker import Faker


@dataclass
class Persona:
    key: str
    role_family: str
    title: str
    seniority: str
    years_experience: int
    city: str
    country: str
    locale: str
    gender: str
    age: int
    stack: list[str]
    languages: list[tuple[str, str]]
    education_hint: str
    company_hint: str
    template: str
    appearance: str = ""
    extra: str = ""
    name_override: str = ""  # for locales where Faker has no Latin-script names
    # filled by `identity()`
    full_name: str = field(default="", init=False)
    email: str = field(default="", init=False)
    phone: str = field(default="", init=False)
    linkedin: str = field(default="", init=False)
    github: str | None = field(default=None, init=False)

    def identity(self, seed: int) -> "Persona":
        fake = Faker(self.locale)
        fake.seed_instance(seed)
        if self.name_override:
            name = self.name_override
        elif self.locale == "ja_JP":
            first = fake.first_romanized_name_male() if self.gender == "male" else fake.first_romanized_name_female()
            name = f"{first} {fake.last_romanized_name()}"
        else:
            first = fake.first_name_male() if self.gender == "male" else fake.first_name_female()
            name = f"{first} {fake.last_name()}"
        self.full_name = re.sub(r"\b(Dr|Prof|Mr|Mrs|Ms|Ing|Dipl\.-Ing|Herr|Frau|Sr|Sra|Dott)\.?\s+", "", name).strip()
        slug = slugify(self.full_name)
        self.email = f"{slug.replace('-', '.')}@{fake.free_email_domain()}"
        self.phone = fake.numerify(PHONE_FORMATS.get(self.country, "+1 ### ### ####"))
        self.linkedin = f"linkedin.com/in/{slug}"
        wants_github = self.role_family not in {"product", "qa"} or self.seniority in {"senior", "lead"}
        self.github = f"github.com/{slug.split('-')[0]}{fake.random_int(2, 98)}" if wants_github else None
        return self

    @property
    def id(self) -> str:
        return f"{self.key}-{slugify(self.full_name)}"

    def photo_prompt(self) -> str:
        return (
            f"Professional corporate headshot photograph of a {self.age}-year-old {self.gender} "
            f"software professional from {self.country}{', ' + self.appearance if self.appearance else ''}. "
            "Plain light grey studio background, soft natural lighting, business-casual clothing, "
            "looking at the camera with a slight smile, sharp focus, photorealistic, 85mm lens."
        )


PHONE_FORMATS = {
    "Germany": "+49 30 ######%#", "Spain": "+34 6## ### ###", "Poland": "+48 ### ### ###",
    "India": "+91 98### #####", "Netherlands": "+31 6 #### ####", "Portugal": "+351 91# ### ###",
    "Argentina": "+54 9 11 #### ####", "Canada": "+1 416 ### ####", "Armenia": "+374 9# ### ###",
    "United Kingdom": "+44 7### ### ###", "Sweden": "+46 70 ### ## ##", "Brazil": "+55 11 9#### ####",
    "Japan": "+81 90 #### ####", "Mexico": "+52 55 #### ####",
}


def slugify(text: str) -> str:
    import unicodedata

    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


PERSONAS: list[Persona] = [
    Persona("p01", "backend", "Senior Backend Engineer", "senior", 10, "Berlin", "Germany", "de_DE", "male", 34,
            ["Python", "Django", "FastAPI", "PostgreSQL", "Kafka", "Docker", "Kubernetes", "AWS"],
            [("German", "Native"), ("English", "C1")], "MSc Computer Science from a German TU",
            "Berlin scale-ups and one large e-commerce company", "modern", "short beard, glasses"),
    Persona("p02", "ml", "Machine Learning Engineer", "senior", 8, "Madrid", "Spain", "es_ES", "female", 31,
            ["Python", "PyTorch", "scikit-learn", "MLflow", "Airflow", "GCP", "Vertex AI", "SQL"],
            [("Spanish", "Native"), ("English", "C1"), ("French", "B1")], "MSc in Data Science, BSc Mathematics",
            "a bank, a travel marketplace and an ML consultancy", "classic", "long dark hair"),
    Persona("p03", "frontend", "Frontend Developer", "junior", 2, "Warsaw", "Poland", "pl_PL", "female", 23,
            ["React", "TypeScript", "Next.js", "Tailwind CSS", "Jest", "Git", "Figma"],
            [("Polish", "Native"), ("English", "B2")], "BSc Computer Science, in progress or just finished",
            "one internship and one product startup", "minimal", "blonde hair"),
    Persona("p04", "data", "Data Engineer", "middle", 5, "Bengaluru", "India", "en_IN", "male", 28,
            ["Apache Spark", "Scala", "Python", "Airflow", "AWS", "Snowflake", "dbt", "Kafka"],
            [("English", "C2"), ("Hindi", "Native"), ("Kannada", "Native")], "BTech from an NIT or IIT",
            "an IT services giant then a fintech", "classic"),
    Persona("p05", "devops", "Lead DevOps / SRE", "lead", 14, "Amsterdam", "Netherlands", "nl_NL", "male", 39,
            ["Kubernetes", "Terraform", "Go", "AWS", "GCP", "Prometheus", "Grafana", "ArgoCD", "Linux", "Bash"],
            [("Dutch", "Native"), ("English", "C2")], "HBO Bachelor in Informatics",
            "a telecom, a payments company and a cloud-native scale-up", "modern", "shaved head"),
    Persona("p06", "qa", "QA Automation Engineer", "middle", 4, "Lisbon", "Portugal", "pt_PT", "female", 29,
            ["Java", "Selenium", "Playwright", "TestNG", "REST Assured", "Postman", "Jenkins", "SQL"],
            [("Portuguese", "Native"), ("Spanish", "B2"), ("English", "C1")], "BSc Software Engineering",
            "an outsourcing firm and a SaaS company", "minimal", "curly hair"),
    Persona("p07", "mobile", "iOS Engineer", "senior", 9, "Buenos Aires", "Argentina", "es_AR", "male", 33,
            ["Swift", "SwiftUI", "UIKit", "Combine", "Core Data", "XCTest", "Fastlane", "REST", "GraphQL"],
            [("Spanish", "Native"), ("English", "B2")], "Licenciatura en Sistemas from UBA",
            "a LatAm fintech and remote US clients", "classic"),
    Persona("p08", "data-science", "Data Scientist", "middle", 4, "Toronto", "Canada", "en_CA", "female", 27,
            ["Python", "pandas", "scikit-learn", "SQL", "R", "Tableau", "A/B testing", "causal inference", "dbt"],
            [("English", "Native"), ("French", "C1"), ("Mandarin", "B1")], "MSc Statistics from a Canadian university",
            "a retail chain analytics team and a health-tech startup", "modern"),
    Persona("p09", "backend", "Python Developer", "junior", 1, "Yerevan", "Armenia", "hy_AM", "male", 22,
            ["Python", "FastAPI", "SQLAlchemy", "PostgreSQL", "Redis", "Docker", "pytest", "Git"],
            [("Armenian", "Native"), ("Russian", "C1"), ("English", "B1")], "BSc from Yerevan State University or NPUA",
            "a bootcamp-style internship and a local outsourcing company", "minimal", name_override="Davit Petrosyan"),
    Persona("p10", "ml", "Principal NLP Researcher", "principal", 15, "London", "United Kingdom", "en_GB", "female", 41,
            ["Python", "PyTorch", "JAX", "Transformers", "LLM fine-tuning", "RLHF", "distributed training", "CUDA"],
            [("English", "Native"), ("German", "B2")], "PhD in Computational Linguistics from Edinburgh or Cambridge",
            "a university lab, a big-tech research group and an AI startup", "classic", "grey-streaked hair",
            "Include 2-3 publications or patents in certifications-style form."),
    Persona("p11", "product", "Senior Product Manager", "senior", 11, "Stockholm", "Sweden", "sv_SE", "male", 36,
            ["Product discovery", "SQL", "Amplitude", "Jira", "Figma", "A/B testing", "OKRs", "Roadmapping"],
            [("Swedish", "Native"), ("English", "C2"), ("Spanish", "A2")], "MSc Industrial Engineering from KTH",
            "a music streaming company and a B2B SaaS", "modern"),
    Persona("p12", "backend", "Java Backend Engineer", "middle", 6, "São Paulo", "Brazil", "pt_BR", "female", 30,
            ["Java", "Spring Boot", "Kotlin", "Hibernate", "PostgreSQL", "RabbitMQ", "Docker", "Azure"],
            [("Portuguese", "Native"), ("English", "B2"), ("Spanish", "B1")], "BSc Computer Engineering from USP or Unicamp",
            "a large Brazilian bank and a delivery app", "classic"),
    Persona("p13", "backend", "Go Engineer", "senior", 9, "Tokyo", "Japan", "ja_JP", "male", 35,
            ["Go", "gRPC", "PostgreSQL", "Redis", "Kubernetes", "AWS", "Terraform", "Protocol Buffers"],
            [("Japanese", "Native"), ("English", "B2")], "BEng from a Japanese national university",
            "a Japanese e-commerce giant and a crypto exchange", "minimal"),
    Persona("p14", "security", "Application Security Engineer", "middle", 5, "Mexico City", "Mexico", "es_MX", "female", 32,
            ["Python", "Bash", "Burp Suite", "OWASP", "SAST/DAST", "AWS security", "Threat modelling", "Semgrep"],
            [("Spanish", "Native"), ("English", "C1")], "BSc Computer Science from UNAM, OSCP",
            "a cybersecurity consultancy and a neobank", "modern"),
]


def personas() -> list[Persona]:
    return [p.identity(seed=1000 + i) for i, p in enumerate(PERSONAS)]
