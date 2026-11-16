__all__ = [
    'CheckReport',
    'Finding',
    'ERROR',
    'WARNING',
    'INFO',
]

from dataclasses import dataclass, field

ERROR = 'ERROR'
WARNING = 'WARNING'
INFO = 'INFO'

_LEVEL_ORDER = {ERROR: 0, WARNING: 1, INFO: 2}


@dataclass
class Finding:
    level: str
    check: str
    message: str
    count: int = 0
    examples: list = field(default_factory=list)

    def to_dict(self):
        return {'level': self.level, 'check': self.check, 'message': self.message,
                'count': self.count, 'examples': [str(e) for e in self.examples]}


class CheckReport:
    MAX_EXAMPLES = 10

    def __init__(self):
        self.findings = []
        self.passed = []

    def add(self, level, check, message, examples=None, count=None):
        examples = list(examples) if examples is not None else []
        self.findings.append(Finding(level, check, message,
                                     count=len(examples) if count is None else count,
                                     examples=examples[:self.MAX_EXAMPLES]))

    def error(self, check, message, examples=None, count=None):
        self.add(ERROR, check, message, examples, count)

    def warning(self, check, message, examples=None, count=None):
        self.add(WARNING, check, message, examples, count)

    def info(self, check, message, examples=None, count=None):
        self.add(INFO, check, message, examples, count)

    def ok(self, check):
        self.passed.append(check)

    def missing(self, check, message, missing_values, level=ERROR):
        """Record a finding if ``missing_values`` is non-empty, else mark ``check`` as passed."""
        missing_values = sorted(missing_values)
        if missing_values:
            self.add(level, check, message, missing_values, count=len(missing_values))
        else:
            self.ok(check)

    @property
    def errors(self):
        return [f for f in self.findings if f.level == ERROR]

    @property
    def warnings(self):
        return [f for f in self.findings if f.level == WARNING]

    def to_dict(self):
        return {
            'errors': len(self.errors),
            'warnings': len(self.warnings),
            'passed': sorted(set(self.passed)),
            'findings': [f.to_dict() for f in self.sorted_findings()],
        }

    def sorted_findings(self):
        return sorted(self.findings, key=lambda f: (_LEVEL_ORDER[f.level], f.check))

    def format(self):
        lines = []
        for f in self.sorted_findings():
            count = f' ({f.count})' if f.count else ''
            lines.append(f'[{f.level}] {f.check}: {f.message}{count}')
            if f.examples:
                more = f' ... +{f.count - len(f.examples)} more' if f.count > len(f.examples) else ''
                lines.append(f'    e.g. {", ".join(str(e) for e in f.examples)}{more}')
        lines.append(f'{len(set(self.passed))} checks passed, {len(self.warnings)} warnings, {len(self.errors)} errors')
        return '\n'.join(lines)
