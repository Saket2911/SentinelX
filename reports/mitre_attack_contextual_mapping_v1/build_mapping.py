import csv
from pathlib import Path
import pandas as pd

OUT = Path("Reports/mitre_attack_contextual_mapping_v1")
SPLITS = {
    "train": Path("Datasets/sentinelx_train_metadata_seq10_v3.csv"),
    "validation": Path("Datasets/sentinelx_validation_metadata_seq10_v3.csv"),
    "test": Path("Datasets/sentinelx_test_metadata_seq10_v3.csv"),
}

# Mapping is contextual, not ground-truth ATT&CK annotation.
MAPPINGS = [
    {
        "label": "FTP-BruteForce",
        "tactic": "Credential Access",
        "technique": "T1110.001",
        "name": "Password Guessing",
        "status": "Conditional candidate",
        "evidence": "Authentication logs showing repeated FTP password guesses.",
        "notes": "The label identifies FTP brute-force activity; confirm the exact behavior with authentication evidence.",
        "source": "https://attack.mitre.org/techniques/T1110/001/",
    },
    {
        "label": "SSH-Bruteforce",
        "tactic": "Credential Access",
        "technique": "T1110.001",
        "name": "Password Guessing",
        "status": "Conditional candidate",
        "evidence": "Authentication logs showing repeated SSH password guesses.",
        "notes": "The label identifies SSH brute-force activity; confirm the exact behavior with authentication evidence.",
        "source": "https://attack.mitre.org/techniques/T1110/001/",
    },
    {
        "label": "Brute Force -Web",
        "tactic": "Credential Access",
        "technique": "T1110.001",
        "name": "Password Guessing",
        "status": "Conditional candidate",
        "evidence": "Web authentication logs showing repeated password guesses.",
        "notes": "Only applies if this label represents web-login password guessing.",
        "source": "https://attack.mitre.org/techniques/T1110/001/",
    },
    {
        "label": "DDoS attacks-LOIC-HTTP",
        "tactic": "Impact",
        "technique": "T1498",
        "name": "Network Denial of Service",
        "status": "Broad candidate",
        "evidence": "Traffic-volume and target-availability evidence; packet-level details for a sub-technique.",
        "notes": "T1498.001 is conditional on confirming a direct network flood.",
        "source": "https://attack.mitre.org/techniques/T1498/",
    },
    {
        "label": "DDOS attack-HOIC",
        "tactic": "Impact",
        "technique": "T1498",
        "name": "Network Denial of Service",
        "status": "Broad candidate",
        "evidence": "Traffic-volume and target-availability evidence; packet-level details for a sub-technique.",
        "notes": "Do not assign a specific sub-technique from the dataset label alone.",
        "source": "https://attack.mitre.org/techniques/T1498/",
    },
    {
        "label": "DDOS attack-LOIC-UDP",
        "tactic": "Impact",
        "technique": "T1498",
        "name": "Network Denial of Service",
        "status": "Broad candidate",
        "evidence": "Traffic-volume and target-availability evidence; packet-level details for a sub-technique.",
        "notes": "T1498.001 is conditional on confirming a direct network flood.",
        "source": "https://attack.mitre.org/techniques/T1498/",
    },
    {
        "label": "DoS attacks-GoldenEye",
        "tactic": "Impact",
        "technique": "T1498",
        "name": "Network Denial of Service",
        "status": "Broad candidate",
        "evidence": "Traffic-volume and service-availability evidence.",
        "notes": "T1499.002 is conditional on evidence of web-service resource exhaustion.",
        "source": "https://attack.mitre.org/techniques/T1499/002/",
    },
    {
        "label": "DoS attacks-Hulk",
        "tactic": "Impact",
        "technique": "T1498",
        "name": "Network Denial of Service",
        "status": "Broad candidate",
        "evidence": "Traffic-volume and service-availability evidence.",
        "notes": "T1499.002 is conditional on evidence of web-service resource exhaustion.",
        "source": "https://attack.mitre.org/techniques/T1499/002/",
    },
    {
        "label": "DoS attacks-SlowHTTPTest",
        "tactic": "Impact",
        "technique": "T1499.002",
        "name": "Service Exhaustion Flood",
        "status": "Conditional candidate",
        "evidence": "Evidence that HTTP service resources were exhausted.",
        "notes": "The label suggests an HTTP DoS scenario but does not independently prove service exhaustion.",
        "source": "https://attack.mitre.org/techniques/T1499/002/",
    },
    {
        "label": "DoS attacks-Slowloris",
        "tactic": "Impact",
        "technique": "T1499.002",
        "name": "Service Exhaustion Flood",
        "status": "Conditional candidate",
        "evidence": "Evidence that web-server resources or connection capacity were exhausted.",
        "notes": "Confirm the effect on the target service before treating this as a technique annotation.",
        "source": "https://attack.mitre.org/techniques/T1499/002/",
    },
    {
        "label": "SQL Injection",
        "tactic": "Initial Access",
        "technique": "T1190",
        "name": "Exploit Public-Facing Application",
        "status": "Conditional candidate",
        "evidence": "Evidence of SQL injection exploiting an Internet-facing application to gain access.",
        "notes": "SQL injection alone does not prove the application was public-facing or that initial access occurred.",
        "source": "https://attack.mitre.org/techniques/T1190/",
    },
    {
        "label": "Bot",
        "tactic": "Command and Control",
        "technique": "",
        "name": "",
        "status": "Insufficient evidence",
        "evidence": "Protocol-level or host-level evidence of command-and-control communication.",
        "notes": "T1071 may be considered only after identifying application-protocol C2. The Bot label alone is insufficient.",
        "source": "https://attack.mitre.org/techniques/T1071/",
    },
    {
        "label": "Infilteration",
        "tactic": "",
        "technique": "",
        "name": "",
        "status": "Insufficient evidence",
        "evidence": "Scenario documentation and host/network evidence describing the intrusion activity.",
        "notes": "Do not interpret this label as exfiltration or assign T1041 without evidence of data transfer over a C2 channel.",
        "source": "https://attack.mitre.org/techniques/T1041/",
    },
    {
        "label": "Brute Force -XSS",
        "tactic": "",
        "technique": "",
        "name": "",
        "status": "Insufficient evidence",
        "evidence": "Application logs and payload/context evidence identifying the actual behavior.",
        "notes": "Do not infer JavaScript execution or assign a technique from this label alone.",
        "source": "",
    },
    {
        "label": "MIXED_ATTACK",
        "tactic": "",
        "technique": "",
        "name": "",
        "status": "Not directly mapped",
        "evidence": "Separate evidence identifying the individual behaviors in the mixed interval.",
        "notes": "A composite state label, not a single ATT&CK technique.",
        "source": "",
    },
    {
        "label": "NO_FLOWS",
        "tactic": "",
        "technique": "",
        "name": "",
        "status": "Not applicable",
        "evidence": "",
        "notes": "Represents a state with no flows; it is not an attack label.",
        "source": "",
    },
    {
        "label": "Benign",
        "tactic": "",
        "technique": "",
        "name": "",
        "status": "Not applicable",
        "evidence": "",
        "notes": "Benign network activity; no attack technique assigned.",
        "source": "",
    },
]


# Refined DoS/DDoS mappings based on the dataset scenario descriptions.
# These remain conditional candidates, not confirmed ATT&CK annotations.
REFINEMENTS = {
    "DDoS attacks-LOIC-HTTP": {
        "tactic": "Impact",
        "technique": "T1498.001",
        "name": "Direct Network Flood",
        "status": "Conditional candidate",
        "evidence": "Traffic volume, packet/request rates, source distribution, and target impact.",
        "notes": "LOIC HTTP DDoS is consistent with direct network flooding; validate the exact behavior using traffic and target-impact evidence.",
        "source": "https://attack.mitre.org/techniques/T1498/001/",
    },
    "DDOS attack-HOIC": {
        "tactic": "Impact",
        "technique": "T1498.001",
        "name": "Direct Network Flood",
        "status": "Conditional candidate",
        "evidence": "Traffic volume, packet/request rates, source distribution, and target impact.",
        "notes": "HOIC DDoS is consistent with direct network flooding; validate the exact behavior using traffic and target-impact evidence.",
        "source": "https://attack.mitre.org/techniques/T1498/001/",
    },
    "DDOS attack-LOIC-UDP": {
        "tactic": "Impact",
        "technique": "T1498.001",
        "name": "Direct Network Flood",
        "status": "Conditional candidate",
        "evidence": "UDP packet rates, traffic volume, source distribution, and target impact.",
        "notes": "LOIC UDP DDoS is consistent with direct UDP network flooding; validate the exact behavior using traffic and target-impact evidence.",
        "source": "https://attack.mitre.org/techniques/T1498/001/",
    },
    "DoS attacks-GoldenEye": {
        "tactic": "Impact",
        "technique": "T1499.002",
        "name": "Service Exhaustion Flood",
        "status": "Conditional candidate",
        "evidence": "HTTP request patterns and server-side resource or service-availability evidence.",
        "notes": "The web DoS scenario is consistent with service exhaustion, but endpoint resource exhaustion is not proven by the label alone.",
        "source": "https://attack.mitre.org/techniques/T1499/002/",
    },
    "DoS attacks-Hulk": {
        "tactic": "Impact",
        "technique": "T1499.002",
        "name": "Service Exhaustion Flood",
        "status": "Conditional candidate",
        "evidence": "HTTP request patterns and server-side resource or service-availability evidence.",
        "notes": "The web DoS scenario is consistent with service exhaustion, but endpoint resource exhaustion is not proven by the label alone.",
        "source": "https://attack.mitre.org/techniques/T1499/002/",
    },
}

for mapping in MAPPINGS:
    if mapping["label"] in REFINEMENTS:
        mapping.update(REFINEMENTS[mapping["label"]])


# Complete evidence and source fields for non-mapped labels.
# These entries deliberately do not assign ATT&CK techniques.
NON_MAPPED_COMPLETIONS = {
    "Benign": {
        "evidence": "Not applicable: benign network activity is not an attack.",
        "source": "https://www.unb.ca/cic/datasets/ids-2018.html",
    },
    "Brute Force -XSS": {
        "evidence": (
            "Application logs and payload/context evidence identifying "
            "the actual behavior. The dataset label alone does not establish "
            "JavaScript execution or a specific ATT&CK technique."
        ),
        "source": "https://www.unb.ca/cic/datasets/ids-2018.html",
    },
    "MIXED_ATTACK": {
        "evidence": (
            "Separate evidence identifying the individual behaviors "
            "within the mixed interval. A single composite label cannot "
            "establish one ATT&CK technique."
        ),
        "source": "https://www.unb.ca/cic/datasets/ids-2018.html",
    },
    "NO_FLOWS": {
        "evidence": (
            "Not applicable: this represents a network state with no "
            "observed flows, not an attack."
        ),
        "source": "https://www.unb.ca/cic/datasets/ids-2018.html",
    },
}

for mapping in MAPPINGS:
    if mapping["label"] in NON_MAPPED_COMPLETIONS:
        mapping.update(NON_MAPPED_COMPLETIONS[mapping["label"]])

# Read actual label counts from all three sequence metadata files.
counts = {}
all_labels = set()

for split, path in SPLITS.items():
    if not path.exists():
        raise FileNotFoundError(f"Missing metadata file: {path}")

    labels = pd.read_csv(path, usecols=["Target Label"])["Target Label"]
    counts[split] = labels.value_counts(dropna=False).to_dict()
    all_labels.update(labels.dropna().unique())

mapping_labels = {row["label"] for row in MAPPINGS}
if all_labels != mapping_labels:
    raise ValueError(
        "Mapping inventory does not match metadata labels.\n"
        f"Missing mappings: {sorted(all_labels - mapping_labels)}\n"
        f"Unexpected mappings: {sorted(mapping_labels - all_labels)}"
    )

csv_path = OUT / "mitre_attack_contextual_mapping_v1.csv"
fields = [
    "label", "train_count", "validation_count", "test_count",
    "tactic", "technique", "name", "status", "evidence",
    "notes", "source",
]

with csv_path.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fields)
    writer.writeheader()

    for row in sorted(MAPPINGS, key=lambda item: item["label"]):
        writer.writerow({
            **row,
            "train_count": counts["train"].get(row["label"], 0),
            "validation_count": counts["validation"].get(row["label"], 0),
            "test_count": counts["test"].get(row["label"], 0),
        })

report_path = OUT / "dataset_label_coverage_report.txt"
with report_path.open("w", encoding="utf-8") as f:
    f.write("SENTINELX V3 — DATASET LABEL COVERAGE AND MITRE MAPPING\n")
    f.write("=" * 65 + "\n\n")
    f.write("Label counts by chronological partition\n\n")

    for split in ("train", "validation", "test"):
        f.write(f"{split.upper()}\n")
        f.write("-" * 30 + "\n")
        for label, count in sorted(counts[split].items()):
            f.write(f"{label}: {count}\n")
        f.write("\n")

    f.write("Important evaluation limitation\n")
    f.write("-" * 30 + "\n")
    f.write(
        "The chronological partitions have different label coverage. "
        "Training contains multiple attack families, validation contains "
        "Benign, Infilteration and NO_FLOWS, while test contains Bot, "
        "Benign and NO_FLOWS. Bot is absent from training. Therefore, "
        "the test partition does not evaluate every training attack family. "
        "This mapping is contextual analyst guidance, not model output "
        "or confirmed ground-truth ATT&CK annotation.\n"
    )

print("Created:")
print(csv_path)
print(report_path)
print(f"\nUnique labels verified: {len(all_labels)}")
print("Mapping inventory matches all three metadata partitions.")
