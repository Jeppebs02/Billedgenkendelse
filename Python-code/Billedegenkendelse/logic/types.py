from __future__ import annotations
from dataclasses import dataclass, asdict
from enum import Enum
from typing import Any, Dict, List, Optional


class Severity(str, Enum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


class Requirement(str, Enum):
    FACE_PRESENT = "FACE_PRESENT"
    SINGLE_FACE = "SINGLE_FACE"
    LANDMARKS_PRESENT = "LANDMARKS_PRESENT"
    EYES_VISIBLE = "EYES_VISIBLE"
    MOUTH_CLOSED = "MOUTH_CLOSED"
    # Vi kan tilføjere flere krav senere som fx CENTERED, EYES_VISIBLE, MOUTH_CLOSED, BG_WHITE,


@dataclass
class CheckResult:
    requirement: Requirement
    passed: bool
    severity: Severity
    message: str
    details: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["requirement"] = self.requirement.value
        d["severity"] = self.severity.value
        return d


@dataclass
class AnalysisReport:
    image: str
    passed: bool
    checks: List[CheckResult]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "image": self.image,
            "passed": self.passed,
            "checks": [c.to_dict() for c in self.checks],
        }

    # Denne printer til console:
    def print_console(self) -> None:
        print(f"Analysis for {self.image} -> {'PASS' if self.passed else 'FAIL'}")
        for c in self.checks:
            status = "✔" if c.passed else "✖"
            print(f"  {status} [{c.severity.value.upper()}] {c.requirement.value}: {c.message}")