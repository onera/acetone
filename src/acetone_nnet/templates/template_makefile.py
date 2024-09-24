from dataclasses import dataclass, field
from typing import ClassVar

from pystache import TemplateSpec


@dataclass
class TemplateMakefile(TemplateSpec):
    """Template for default Makefile script."""

    template_extension: ClassVar[str] = "tpl"

    executable_name: str
    compiler: str = field(default="gcc")
    compiler_flags: list[str] = field(default_factory=list)
    source_files: list[str] = field(default_factory=list)
    header_files: list[str] = field(default_factory=list)

    def add_source_files(self, *filenames: str) -> None:
        for f in filenames:
            if f not in self.source_files:
                self.source_files.append(f)

    def add_header_files(self, *filenames: str) -> None:
        for f in filenames:
            if f not in self.header_files:
                self.header_files.append(f)

    def add_compiler_flags(self, *flags: str) -> None:
        self.compiler_flags.extend(flags)
