"""
Interactive and command-line interface for the product workflow.

This module validates input/output paths, gathers user parameters,
and invokes payload encoding/decoding with encryption from payload_manager
and embedding/extraction via the stegano_core backend. It also exposes
inspection utilities that generate visual diagnostics for stego pairs.
"""

from __future__ import annotations

import argparse
import re
import platform
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import psutil
from cryptography.exceptions import InvalidTag
from rich.console import Console
from rich.columns import Columns
from rich.prompt import Confirm, IntPrompt, Prompt
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.theme import Theme
from rich import box

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from stegano_app.payload_manager import decode_bits, encode_file, encode_message
from stegano_app.product_visualize import inspect_pair

import stegano_core


ACCENT = "#f7bf02"
ACCENT_DIM = "#c17b00"
HILITE = "#ff9a00"
MUTED = "#6f5a2a"
ERROR = "#ff4d4d"

THEME = Theme(
    {
        "accent": f"bold {ACCENT}",
        "accent_dim": ACCENT_DIM,
        "hilite": HILITE,
        "muted": MUTED,
        "ok": f"bold {ACCENT}",
        "warn": f"bold {HILITE}",
        "error": f"bold {ERROR}",
    }
)


def _load_image(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"could not read image: {path}")
    if img.ndim == 3:
        if img.shape[2] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        elif img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
    return img.astype(np.uint8, copy=False)


def _save_image(path: Path, img: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    save_img = img
    if img.ndim == 3:
        save_img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    ok = cv2.imwrite(str(path), save_img)
    if not ok:
        raise OSError(f"failed to write image: {path}")


def _ensure_readable_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"input file not found: {path}")
    if not path.is_file():
        raise FileNotFoundError(f"input path is not a file: {path}")


def _ensure_output_path(path: Path, force: bool) -> None:
    if path.exists() and not force:
        raise FileExistsError(
            f"output already exists: {path} (use --force to overwrite)"
        )


def _require_prompt_toolkit(console: Console) -> None:
    try:
        import prompt_toolkit  # noqa: F401
    except ModuleNotFoundError as exc:
        console.print(
            "[red]Interactive mode requires prompt_toolkit for path completion.[/red]\n"
            "Install it with: [bold]python -m pip install prompt_toolkit[/bold]"
        )
        raise exc


def _require_questionary(console: Console) -> None:
    try:
        import questionary  # noqa: F401
        import questionary.prompts.common
        
        def _patched_create_layout(ic, get_prompt_tokens, **kwargs):
            bottom_toolbar = kwargs.pop("bottom_toolbar", None)
            from prompt_toolkit.shortcuts import PromptSession
            from prompt_toolkit.layout import Layout
            from prompt_toolkit.layout.containers import HSplit, ConditionalContainer, Window
            from prompt_toolkit.layout.controls import FormattedTextControl
            from prompt_toolkit.layout.dimension import LayoutDimension
            from prompt_toolkit.filters import Condition, IsDone
            
            ps = PromptSession(
                get_prompt_tokens, reserve_space_for_menu=0, **kwargs
            )
            questionary.prompts.common._fix_unecessary_blank_lines(ps)
            
            @Condition
            def has_search_string():
                return ic.get_search_string_tokens() is not None
                
            validation_prompt = PromptSession(
                bottom_toolbar=lambda: ic.error_message, **kwargs
            )
            
            components = [
                ps.layout.container,
                ConditionalContainer(Window(ic), filter=~IsDone()),
                ConditionalContainer(
                    Window(height=LayoutDimension.exact(2), content=FormattedTextControl(ic.get_search_string_tokens)),
                    filter=has_search_string & ~IsDone(),
                ),
                ConditionalContainer(
                    validation_prompt.layout.container,
                    filter=Condition(lambda: ic.error_message is not None),
                ),
            ]
            
            if bottom_toolbar is not None:
                toolbar_content = bottom_toolbar() if callable(bottom_toolbar) else bottom_toolbar
                components.append(
                    ConditionalContainer(
                        Window(content=FormattedTextControl(lambda: toolbar_content), height=LayoutDimension.exact(1), style="class:bottom-toolbar"),
                        filter=~IsDone()
                    )
                )
            
            return Layout(HSplit(components))
        questionary.prompts.common.create_inquirer_layout = _patched_create_layout

    except ModuleNotFoundError as exc:
        console.print(
            "[red]Interactive mode requires questionary for the arrow-key menu.[/red]\n"
            "Install it with: [bold]python -m pip install questionary[/bold]"
        )
        raise exc


def _menu_toolbar() -> list[tuple[str, str]]:
    return [
        ("#ffffff bg:#ff9a00 bold noreverse", " STEGANO "),
        (
            "#b0b0b0 bg:#1a1a1a noreverse",
            "  ↑↓·navigate | enter·select | ctrl+c·exit ",
        ),
    ]

def _input_toolbar() -> list[tuple[str, str]]:
    return [
        ("#ffffff bg:#ff9a00 bold noreverse", " STEGANO "),
        (
            "#b0b0b0 bg:#1a1a1a noreverse",
            "  enter·submit | ctrl+c·back to menu  ",
        ),
    ]

def _prompt_path(
    message: str,
    console: Console,
    *,
    default: str | None = None,
    must_exist: bool = True,
) -> Path:
    import questionary
    from questionary import Style

    def _validate(value: str) -> bool | str:
        if not value:
            return "Path is required."
        value = value.strip(" '\"")
        path = Path(value).expanduser()
        if must_exist:
            if not path.exists():
                return f"Input file not found: {path}."
            if not path.is_file():
                return f"Input path is not a file: {path}."
        return True

    style = Style(
        [
            ("qmark", f"fg:{HILITE} bold"),
            ("question", ""),
            ("answer", ""),
        
            ("bottom-toolbar", "#b0b0b0 bg:#1a1a1a noreverse"),
        ]
    )
    answer = questionary.path(
        message,
        default=default or "",
        qmark="> ",
        validate=_validate,
        style=style,
        bottom_toolbar=_input_toolbar(),
    ).ask()
    if answer is None:
        raise KeyboardInterrupt
    return Path(answer.strip(" '\"")).expanduser()


def _prompt_output_path(message: str, default: str, console: Console) -> tuple[Path, bool]:
    import questionary
    from questionary import Style

    style = Style(
        [
            ("qmark", f"fg:{HILITE} bold"),
            ("question", ""),
            ("answer", ""),
        
            ("bottom-toolbar", "#b0b0b0 bg:#1a1a1a noreverse"),
        ]
    )

    while True:
        path = _prompt_path(message, console, default=default, must_exist=False)
        if path.exists():
            answer = questionary.text(
                "File already exists. Overwrite? [y/n]",
                qmark="> ",
                validate=lambda v: True
                if v.lower() in {"y", "yes", "n", "no"}
                else "Please enter y or n.",
                style=style,
                bottom_toolbar=_input_toolbar(),
            ).ask()
            if answer is None:
                raise KeyboardInterrupt
            if answer.lower() in {"y", "yes"}:
                return path, True
            continue
        return path, False


def cmd_lock(ns: argparse.Namespace, console: Console) -> int:
    if (ns.message is None) == (ns.file is None):
        raise ValueError("provide exactly one of --message or --file")

    cover_path = Path(ns.input)
    output_path = Path(ns.output)
    valid_exts = {".pgm", ".png", ".tiff", ".tif", ".bmp", ".jpg", ".jpeg"}
    if cover_path.suffix.lower() not in valid_exts:
        raise ValueError(f"Input must be a supported image file ({', '.join(valid_exts)}).")
    _ensure_readable_file(cover_path)
    _ensure_output_path(output_path, ns.force)

    if ns.file is not None:
        file_path = Path(ns.file)
        _ensure_readable_file(file_path)

    cover = _load_image(cover_path)
    if ns.message is not None:
        payload_bits = encode_message(ns.message, ns.password)
    else:
        payload_bits = encode_file(ns.file, ns.password)

    payload_bits = np.ascontiguousarray(payload_bits, dtype=np.uint8)
    with console.status("Analyzing image and embedding payload...", spinner="dots"):
        if cover.ndim == 2:
            stego = stegano_core.run_product_embed(
                cover,
                payload_bits,
                ns.password,
                ns.colony_size,
                ns.max_iter,
                ns.min_block,
            )
        elif cover.ndim == 3:
            payload_len = len(payload_bits)
            base_len = payload_len // 3
            c_payloads = [
                np.ascontiguousarray(payload_bits[:base_len]),
                np.ascontiguousarray(payload_bits[base_len:2*base_len]),
                np.ascontiguousarray(payload_bits[2*base_len:])
            ]
            stego_channels = []
            for i in range(3):
                c_stego = stegano_core.run_product_embed(
                    np.ascontiguousarray(cover[:, :, i]),
                    c_payloads[i],
                    ns.password,
                    ns.colony_size,
                    ns.max_iter,
                    ns.min_block,
                )
                stego_channels.append(c_stego)
            stego = np.dstack(stego_channels)
    _save_image(output_path, stego)
    console.print(
        Panel.fit(
            f"Saved stego image:\n{output_path}",
            border_style="accent",
            box=box.ASCII,
        )
    )
    return 0


def cmd_unlock(ns: argparse.Namespace, console: Console) -> int:
    stego_path = Path(ns.input)
    output_path = Path(ns.output)
    valid_exts = {".pgm", ".png", ".tiff", ".tif", ".bmp", ".jpg", ".jpeg"}
    if stego_path.suffix.lower() not in valid_exts:
        raise ValueError(f"Input must be a supported image file ({', '.join(valid_exts)}).")
    _ensure_readable_file(stego_path)
    _ensure_output_path(output_path, ns.force)

    stego = _load_image(stego_path)
    with console.status("Extracting and decrypting payload...", spinner="dots"):
        if stego.ndim == 2:
            payload_bits = stegano_core.run_product_extract(
                stego,
                ns.password,
                ns.colony_size,
                ns.max_iter,
                ns.min_block,
            )
        elif stego.ndim == 3:
            c_payloads = []
            for i in range(3):
                bits = stegano_core.run_product_extract(
                    np.ascontiguousarray(stego[:, :, i]),
                    ns.password,
                    ns.colony_size,
                    ns.max_iter,
                    ns.min_block,
                )
                c_payloads.append(bits)
            payload_bits = np.concatenate(c_payloads)
            
        try:
            data = decode_bits(payload_bits, ns.password)
        except (InvalidTag, ValueError) as exc:
            raise ValueError("Invalid password or corrupted payload data.") from exc
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as handle:
        handle.write(data)
    console.print(
        Panel.fit(
            f"Saved extracted payload:\n{output_path}",
            border_style="accent",
            box=box.ASCII,
        )
    )

    is_text = False
    decoded_text = ""
    
    if len(data) == 0:
        is_text = True
    elif b'\x00' not in data:
        # Heuristic: reject files with >5% non-whitespace control characters
        control_chars = sum(1 for b in data if b < 32 and b not in (9, 10, 13))
        if (control_chars / len(data)) < 0.05:
            # Fallback chain prioritizing UTF-8 (with/without BOM), then regional, then catch-all
            for enc in ["utf-8-sig", "utf-8", "cp1254", "cp1252", "latin-1"]:
                try:
                    decoded_text = data.decode(enc)
                    is_text = True
                    break
                except UnicodeDecodeError:
                    continue

    if is_text:
        if getattr(ns, "print", False):
            console.print(Panel(decoded_text, title="[bold accent]Extracted Content[/]", box=box.ROUNDED))
        elif getattr(ns, "interactive_mode", False):
            import questionary
            ans = questionary.text(
                "Payload saved to disk. Readable text detected. Print to stdout? [y/n]",
                qmark="> ",
                validate=lambda v: True if v.lower() in {"y", "yes", "n", "no"} else "Please enter y or n.",
            ).ask()
            if ans and ans.lower() in {"y", "yes"}:
                console.print(Panel(decoded_text, title="[bold accent]Extracted Content[/]", box=box.ROUNDED))
    else:
        if getattr(ns, "print", False):
            console.print("[warn]Warning:[/] The extracted payload is a binary file and cannot be printed to the terminal.")

    return 0


def cmd_inspect(ns: argparse.Namespace, console: Console) -> int:
    cover_path = Path(ns.cover)
    stego_path = Path(ns.stego)
    output_path = Path(ns.output)
    valid_exts = {".pgm", ".png", ".tiff", ".tif", ".bmp", ".jpg", ".jpeg"}
    if cover_path.suffix.lower() not in valid_exts or stego_path.suffix.lower() not in valid_exts:
        raise ValueError(f"Input must be a supported image file ({', '.join(valid_exts)}).")
    _ensure_readable_file(cover_path)
    _ensure_readable_file(stego_path)
    _ensure_output_path(output_path, ns.force)

    cover = _load_image(cover_path)
    stego = _load_image(stego_path)
    if cover.shape != stego.shape:
        raise ValueError("cover and stego images must have the same dimensions")

    with console.status("Rendering inspection figure...", spinner="dots"):
        metrics = inspect_pair(
            cover,
            stego,
            output_path,
            cover_name=cover_path.name,
            stego_name=stego_path.name,
        )

    console.print(
        Panel.fit(
            f"Saved inspection figure:\n{output_path}",
            border_style="green",
        )
    )

    table = Table(show_header=True, header_style="accent", box=box.ASCII)
    table.add_column("Metric")
    table.add_column("Value", justify="right")
    table.add_row("PSNR (dB)", f"{metrics['psnr']:.2f}")
    table.add_row("SSIM", f"{metrics['ssim']:.5f}")
    table.add_row("MSE", f"{metrics['mse']:.4f}")
    table.add_row("Changed Pixels", f"{metrics['changed_pixels']:,}")
    console.print(
        Panel.fit(
            table,
            title="Inspection Metrics",
            border_style="accent",
            box=box.ASCII,
        )
    )
    return 0


def _lock_menu(console: Console) -> argparse.Namespace:
    import questionary
    from questionary import Style

    steps = [
        "Input Path",
        "Output Path",
        "Password",
        "Embed",
        "Text",
        "Parameters",
    ]

    _prompt_style = Style(
        [
            ("qmark", f"fg:{HILITE} bold"),
            ("question", ""),
            ("answer", ""),
            ("instruction", "fg:#888888"),
        
            ("bottom-toolbar", "#b0b0b0 bg:#1a1a1a noreverse"),
        ]
    )

    def _render_step(active_index: int, description: str) -> None:
        console.clear()
        console.print("[bold #ff9a00]Lock[/]")
        console.print(
            Text(
                "Embed a secure, encrypted payload into a cover image using D-ABC optimization.",
                style="dim",
            )
        )
        console.print()
        console.print(Text("─" * 80, style=MUTED))

        breadcrumb = Text()
        for idx, label in enumerate(steps):
            if idx:
                breadcrumb.append(" | ", style=MUTED)
            if idx == active_index:
                breadcrumb.append(label, style=f"bold {ACCENT}")
            else:
                breadcrumb.append(label, style=MUTED)
        console.print(breadcrumb)
        console.print()
        console.print(Text(description, style="dim"))

    def _print_param_value(label: str, default: int, value: int) -> None:
        line = Text()
        line.append("> ", style=f"bold {HILITE}")
        line.append(f"{label} ({default}): ")
        line.append(str(value))
        console.print(line)

    def _prompt_path_value(
        active_index: int,
        description: str,
        label: str,
        *,
        must_exist: bool,
        require_pgm: bool = False,
    ) -> Path:
        def _validate(value: str) -> bool | str:
            if not value:
                return "Path is required."
            value = value.strip(" '\"")
            path = Path(value).expanduser()
            if must_exist:
                if not path.exists():
                    return f"Input file not found: {path}."
                if not path.is_file():
                    return f"Input path is not a file: {path}."
            if require_pgm and path.suffix.lower() not in {".pgm", ".png", ".tiff", ".tif", ".bmp", ".jpg", ".jpeg"}:
                return "Input must be a supported image file (.pgm, .png, .tiff, etc)."
            return True

        _render_step(active_index, description)
        value = questionary.path(
            label,
            qmark="> ",
            validate=_validate,
            style=_prompt_style,
            bottom_toolbar=_input_toolbar(),
        ).ask()
        if value is None:
            raise KeyboardInterrupt
        return Path(value.strip(" '\"")).expanduser()

    def _prompt_password(active_index: int) -> str:
        _render_step(active_index, "Set a strong password for AES-GCM encryption:")
        value = questionary.password(
            "Password",
            qmark="> ",
            validate=lambda v: True if v else "Password is required.",
            style=_prompt_style,
            bottom_toolbar=_input_toolbar(),
        ).ask()
        if value is None:
            raise KeyboardInterrupt
        return value

    def _prompt_message(active_index: int) -> str:
        _render_step(active_index, "Enter the secret text message:")
        value = questionary.text(
            "Message",
            qmark="> ",
            validate=lambda v: True if v else "Message is required.",
            style=_prompt_style,
            bottom_toolbar=_input_toolbar(),
        ).ask()
        if value is None:
            raise KeyboardInterrupt
        return value

    def _prompt_int_value(
        active_index: int,
        description: str,
        label: str,
        default: int,
        prior_values: list[tuple[str, int, int]],
    ) -> int:
        def _validate(value: str) -> bool | str:
            if not value:
                return True
            try:
                parsed = int(value)
            except ValueError:
                return "Enter a positive integer."
            if parsed <= 0:
                return "Enter a positive integer."
            return True

        _render_step(active_index, description)
        for prev_label, prev_default, prev_value in prior_values:
            _print_param_value(prev_label, prev_default, prev_value)
        value = questionary.text(
            f"{label} ({default})",
            qmark="> ",
            validate=_validate,
            style=_prompt_style,
            bottom_toolbar=_input_toolbar(),
        ).ask()
        if value is None:
            raise KeyboardInterrupt
        if not value:
            return default
        return int(value)

    cover = _prompt_path_value(
        0,
        "Enter the path to the original, untouched cover image:",
        "Input Cover Image",
        must_exist=True,
        require_pgm=True,
    )

    while True:
        _render_step(1, "Specify the destination path for the generated stego image:")
        output_value = questionary.path(
            "Output Stego Image",
            qmark="> ",
            validate=lambda v: True if v else "Path is required.",
            style=_prompt_style,
            bottom_toolbar=_input_toolbar(),
        ).ask()
        if output_value is None:
            raise KeyboardInterrupt
        output = Path(output_value.strip(" '\"")).expanduser()
        if output.exists():
            _render_step(1, "Specify the destination path for the generated stego image:")
            console.print(f"> Output Stego Image: {output}")
            answer = questionary.text(
                "File already exists. Overwrite? [y/n]",
                qmark="> ",
                validate=lambda v: True
                if v.lower() in {"y", "yes", "n", "no"}
                else "Please enter y or n.",
                style=_prompt_style,
                bottom_toolbar=_input_toolbar(),
            ).ask()
            if answer is None:
                raise KeyboardInterrupt
            if answer.lower() in {"y", "yes"}:
                force = True
                break
            continue
        force = False
        break

    password = _prompt_password(2)

    _render_step(3, "What type of data do you want to embed?")

    menu_style = Style(
        [
            ("qmark", f"fg:{HILITE} bold"),
            ("pointer", f"fg:{HILITE} bold"),
            ("highlighted", f"fg:{ACCENT} bold"),
            ("text", f"fg:{MUTED}"),
        
            ("bottom-toolbar", "#b0b0b0 bg:#1a1a1a noreverse"),
        ]
    )

    while True:
        prompt_obj = questionary.select(
            "",
            choices=[
                questionary.Choice("message", value="message"),
                questionary.Choice("file", value="file"),
            ],
            style=menu_style,
            qmark="",
            pointer="> ",
            instruction="",
            bottom_toolbar=_menu_toolbar(),
        )
        bindings = prompt_obj.application.key_bindings
        
        @bindings.add("q")
        def _q2(event):
            event.app.exit(result="quit")
            
        @bindings.add("?")
        def _help2(event):
            event.app.exit(result="help")
            
        mode = prompt_obj.ask()
        
        if mode == "quit":
            raise KeyboardInterrupt
        if mode == "help":
            from rich.panel import Panel
            console.print(Panel("Choose [bold]message[/bold] to type a short text payload, or [bold]file[/bold] to embed an entire document.", border_style="accent"))
            continue
        break

    if mode == "message":
        message = _prompt_message(4)
        file_path = None
    else:
        file_path = _prompt_path_value(
            4,
            "Enter the path to the secret file:",
            "Secret File",
            must_exist=True,
        )
        message = None

    _render_step(5, "Configure the Discrete ABC and Quadtree settings (Press Enter for defaults):")

    def _validate_pos_int(v: str) -> bool | str:
        if not v:
            return True
        if not v.isdigit() or int(v) <= 0:
            return "Enter a positive integer."
        return True

    colony_str = questionary.text(
        "Colony Size (30)",
        default="30",
        qmark="> ",
        validate=_validate_pos_int,
        style=_prompt_style,
        bottom_toolbar=_input_toolbar(),
    ).ask()
    if colony_str is None:
        raise KeyboardInterrupt
    colony_size = int(colony_str) if colony_str else 30

    iter_str = questionary.text(
        "Max Iterations (50)",
        default="50",
        qmark="> ",
        validate=_validate_pos_int,
        style=_prompt_style,
        bottom_toolbar=_input_toolbar(),
    ).ask()
    if iter_str is None:
        raise KeyboardInterrupt
    max_iter = int(iter_str) if iter_str else 50

    block_str = questionary.text(
        "Min Blocks (4)",
        default="4",
        qmark="> ",
        validate=_validate_pos_int,
        style=_prompt_style,
        bottom_toolbar=_input_toolbar(),
    ).ask()
    if block_str is None:
        raise KeyboardInterrupt
    min_block = int(block_str) if block_str else 4

    return argparse.Namespace(
        input=str(cover),
        output=str(output),
        password=password,
        message=message,
        file=str(file_path) if file_path else None,
        colony_size=colony_size,
        max_iter=max_iter,
        min_block=min_block,
        force=force,
    )


def _execute_unlock_protocol(console: Console) -> argparse.Namespace:
    import questionary
    from questionary import Style

    steps = [
        "Stego Path",
        "Output Path",
        "Password",
        "Parameters",
    ]

    _prompt_style = Style(
        [
            ("qmark", f"fg:{HILITE} bold"),
            ("question", ""),
            ("answer", ""),
            ("instruction", "fg:#888888"),
            ("bottom-toolbar", "#b0b0b0 bg:#1a1a1a noreverse"),
        ]
    )

    def _render_step(active_index: int, description: str) -> None:
        console.clear()
        console.print("[bold #ff9a00]Unlock[/]")
        console.print(
            Text(
                "Extract a hidden payload from a stego image using D-ABC optimization.",
                style="dim",
            )
        )
        console.print()
        console.print(Text("─" * 80, style=MUTED))
        
        breadcrumb = Text()
        for idx, label in enumerate(steps):
            if idx:
                breadcrumb.append(" | ", style=MUTED)
            if idx == active_index:
                breadcrumb.append(label, style=f"bold {ACCENT}")
            else:
                breadcrumb.append(label, style=MUTED)
        console.print(breadcrumb)
        console.print()
        if description:
            console.print(Text(description, style="dim"))

    def _print_param_value(label: str, default: int, value: int) -> None:
        line = Text()
        line.append("> ", style=f"bold {HILITE}")
        line.append(f"{label} ({default}): ")
        line.append(str(value))
        console.print(line)

    def _prompt_path_value(
        active_index: int,
        description: str,
        label: str,
        *,
        must_exist: bool,
        require_pgm: bool = False,
    ) -> Path:
        def _validate(value: str) -> bool | str:
            if not value:
                return "Path is required."
            value = value.strip(" '\"")
            path = Path(value).expanduser()
            if must_exist:
                if not path.exists():
                    return f"Input file not found: {path}."
                if not path.is_file():
                    return f"Input path is not a file: {path}."
            if require_pgm and path.suffix.lower() not in {".pgm", ".png", ".tiff", ".tif", ".bmp", ".jpg", ".jpeg"}:
                return "Input must be a supported image file (.pgm, .png, .tiff, etc)."
            return True

        _render_step(active_index, description)
        value = questionary.path(
            label,
            qmark="> ",
            validate=_validate,
            style=_prompt_style,
            bottom_toolbar=_input_toolbar(),
        ).ask()
        if value is None:
            raise KeyboardInterrupt
        return Path(value.strip(" '\"")).expanduser()

    def _prompt_output_path_value(
        active_index: int,
        description: str,
        label: str,
        default_file: str,
    ) -> tuple[Path, bool]:
        force = False
        while True:
            _render_step(active_index, description)
            value = questionary.path(
                label,
                default=default_file,
                qmark="> ",
                validate=lambda v: True if v else "Output path is required.",
                style=_prompt_style,
                bottom_toolbar=_input_toolbar(),
            ).ask()
            if value is None:
                raise KeyboardInterrupt
            
            output = Path(value.strip(" '\"")).expanduser()
            if output.exists():
                ans = questionary.text(
                    "File already exists. Overwrite? [y/n]",
                    qmark="> ",
                    validate=lambda v: True if v.lower() in {"y", "yes", "n", "no"} else "Please enter y or n.",
                    style=_prompt_style,
                    bottom_toolbar=_input_toolbar(),
                ).ask()
                if ans is None:
                    raise KeyboardInterrupt
                if ans.lower() in {"y", "yes"}:
                    force = True
                    return output, force
                # else continue loop
            else:
                return output, force

    def _prompt_password_value(active_index: int) -> str:
        _render_step(active_index, "Enter the password used to encrypt the payload.")
        value = questionary.password(
            "Enter Password",
            qmark="> ",
            validate=lambda v: True if v else "Password is required.",
            style=_prompt_style,
            bottom_toolbar=_input_toolbar(),
        ).ask()
        if value is None:
            raise KeyboardInterrupt
        return value

    def _prompt_params_value(active_index: int) -> tuple[int, int, int]:
        _render_step(active_index, "Configure the D-ABC extraction parameters.")

        def _validate_pos_int(v: str) -> bool | str:
            if not v.isdigit() or int(v) <= 0:
                return "Please enter a valid positive integer."
            return True

        colony_str = questionary.text(
            "Colony Size (30)",
            qmark="> ",
            default="30",
            validate=_validate_pos_int,
            style=_prompt_style,
            bottom_toolbar=_input_toolbar(),
        ).ask()
        if colony_str is None:
            raise KeyboardInterrupt
        c_val = int(colony_str)

        iter_str = questionary.text(
            "Max Iterations (50)",
            qmark="> ",
            default="50",
            validate=_validate_pos_int,
            style=_prompt_style,
            bottom_toolbar=_input_toolbar(),
        ).ask()
        if iter_str is None:
            raise KeyboardInterrupt
        i_val = int(iter_str)

        block_str = questionary.text(
            "Min Blocks (4)",
            qmark="> ",
            default="4",
            validate=_validate_pos_int,
            style=_prompt_style,
            bottom_toolbar=_input_toolbar(),
        ).ask()
        if block_str is None:
            raise KeyboardInterrupt
        b_val = int(block_str)

        return c_val, i_val, b_val

    stego_path = _prompt_path_value(
        0,
        "Specify the path to the stego image containing the hidden payload:",
        "Enter Stego Path",
        must_exist=True,
        require_pgm=True,
    )

    output_path, force = _prompt_output_path_value(
        1,
        "Specify the destination path for the extracted payload:",
        "Enter Output Path",
        "extracted.bin",
    )

    password = _prompt_password_value(2)

    colony, iterations, blocks = _prompt_params_value(3)

    return argparse.Namespace(
        input=str(stego_path),
        output=str(output_path),
        password=password,
        colony_size=colony,
        max_iter=iterations,
        min_block=blocks,
        force=force,
        interactive_mode=True,
    )


def _wizard_inspect(console: Console) -> argparse.Namespace:
    import questionary
    from questionary import Style

    steps = ["Cover Path", "Stego Path", "Output Path"]

    _prompt_style = Style(
        [
            ("qmark", f"fg:{HILITE} bold"),
            ("question", ""),
            ("answer", ""),
            ("instruction", "fg:#888888"),
            ("bottom-toolbar", "#b0b0b0 bg:#1a1a1a noreverse"),
        ]
    )

    def _render_step(active_index: int, description: str) -> None:
        console.clear()
        console.print("[bold #ff9a00]Inspect[/]")
        console.print(
            Text(
                "Render a publication-ready visual analysis comparing cover and stego images.",
                style="dim",
            )
        )
        console.print()
        console.print(Text("─" * 80, style=MUTED))
        
        breadcrumb = Text()
        for idx, label in enumerate(steps):
            if idx:
                breadcrumb.append(" | ", style=MUTED)
            if idx == active_index:
                breadcrumb.append(label, style=f"bold {ACCENT}")
            else:
                breadcrumb.append(label, style=MUTED)
        console.print(breadcrumb)
        console.print()
        if description:
            console.print(Text(description, style="dim"))

    def _prompt_path_value(
        active_index: int,
        description: str,
        label: str,
        *,
        must_exist: bool,
        require_pgm: bool = False,
    ) -> Path:
        def _validate(value: str) -> bool | str:
            if not value:
                return "Path is required."
            value = value.strip(" '\"")
            path = Path(value).expanduser()
            if must_exist:
                if not path.exists():
                    return f"Input file not found: {path}."
                if not path.is_file():
                    return f"Input path is not a file: {path}."
            if require_pgm and path.suffix.lower() not in {".pgm", ".png", ".tiff", ".tif", ".bmp", ".jpg", ".jpeg"}:
                return "Input must be a supported image file (.pgm, .png, .tiff, etc)."
            return True

        _render_step(active_index, description)
        value = questionary.path(
            label,
            qmark="> ",
            validate=_validate,
            style=_prompt_style,
            bottom_toolbar=_input_toolbar(),
        ).ask()
        if value is None:
            raise KeyboardInterrupt
        return Path(value.strip(" '\"")).expanduser()

    def _prompt_output_path_value(
        active_index: int,
        description: str,
        label: str,
        default_file: str,
    ) -> tuple[Path, bool]:
        force = False
        while True:
            _render_step(active_index, description)
            value = questionary.path(
                label,
                default=default_file,
                qmark="> ",
                validate=lambda v: True if v else "Output path is required.",
                style=_prompt_style,
                bottom_toolbar=_input_toolbar(),
            ).ask()
            if value is None:
                raise KeyboardInterrupt
            
            output = Path(value.strip(" '\"")).expanduser()
            if output.exists():
                ans = questionary.text(
                    "File already exists. Overwrite? [y/n]",
                    qmark="> ",
                    validate=lambda v: True if v.lower() in {"y", "yes", "n", "no"} else "Please enter y or n.",
                    style=_prompt_style,
                    bottom_toolbar=_input_toolbar(),
                ).ask()
                if ans is None:
                    raise KeyboardInterrupt
                if ans.lower() in {"y", "yes"}:
                    force = True
                    return output, force
            else:
                return output, force

    cover = _prompt_path_value(
        0,
        "Enter the path to the original, untouched cover image:",
        "Cover Image Path",
        must_exist=True,
        require_pgm=True,
    )

    stego = _prompt_path_value(
        1,
        "Enter the path to the stego image to be analyzed:",
        "Stego Image Path",
        must_exist=True,
        require_pgm=True,
    )

    output, force = _prompt_output_path_value(
        2,
        "Specify the destination path for the generated analysis figure:",
        "Output Figure Path",
        "inspection_result.png",
    )

    return argparse.Namespace(
        cover=str(cover),
        stego=str(stego),
        output=str(output),
        force=force,
    )

def purify_ansi_banner(raw_ansi: str) -> str:
    """libcaca'nın kirli ANSI çıktısını temizler."""
    
    ansi = raw_ansi.replace(";5;", ";").replace("[0;5;", "[0;")
    
    ansi = ansi.replace(".", " ")
    ansi = ansi.replace("47m", "49m")
    ansi = ansi.replace("37m", "39m")

    ansi = ansi.replace("31m", "30m") 
    ansi = ansi.replace("41m", "40m") 
    

    for bad_bg in ["42m", "44m", "45m", "46m"]:
        ansi = ansi.replace(bad_bg, "40m")
    for bad_fg in ["32m", "34m", "35m", "36m"]:
        ansi = ansi.replace(bad_fg, "30m")
    ansi = re.sub(r"[ \t]+\x1b\[0m", "\x1b[0m", ansi)

    return ansi


def _print_branding(console: Console) -> None:
    console.clear()
    banner_path = Path(__file__).parent / "STEGANO_1.ansi.txt"

    banner_header = Text()
    banner_header.append("> ", style=f"bold {HILITE}")
    banner_header.append("stegano\n\n", style="bold white")

    try:
        raw_ansi = banner_path.read_text(encoding="utf-8")
        clean_ansi = purify_ansi_banner(raw_ansi)
        banner_body = Text.from_ansi(clean_ansi, no_wrap=True)
        banner_obj = Text.assemble(banner_header, banner_body)
        banner_obj.no_wrap = True
        banner_lines = banner_obj.plain.splitlines()
    except FileNotFoundError:
        banner_obj = Text.assemble(banner_header, Text("STEGANO", style=ACCENT))
        banner_obj.no_wrap = True
        banner_lines = banner_obj.plain.splitlines()

    bee_height = len(banner_lines)

    specs = Table(show_header=False, box=None, show_edge=False, padding=(0, 4))
    
    specs.add_column(justify="left", style=ACCENT, no_wrap=True, overflow="ignore")
    specs.add_column(justify="left", style="white", no_wrap=True, overflow="ignore")
    
    specs.add_row("Core", "Rust/PyO3")
    specs.add_row("Method", "ABC/LSB-M")
    specs.add_row("Filter", "Quadtree")
    specs.add_row("Cipher", "AES-GCM")

    right_panel = Panel(
        specs,
        title=f"[bold {ACCENT}]STEGANO[/]",
        subtitle="[dim]v0.2.1[/dim]",
        subtitle_align="right",
        border_style="#444444",
        box=box.ROUNDED,
        padding=(0, 1),
    )

    layout = Table.grid(padding=(0, 1))
    layout.add_column(no_wrap=True, justify="left", vertical="middle")
    layout.add_column(no_wrap=True, justify="center", vertical="middle")
    layout.add_row(banner_obj, right_panel)

    console.print(layout)

def _show_help(console: Console) -> None:
    console.clear()
    
    # DESCRIPTION
    desc_text = Text("STEGANO is a secure data embedding tool utilizing AES-GCM encryption, Adaptive Quadtree for texture analysis, and Discrete Artificial Bee Colony (D-ABC) optimization via LSB-Matching.", style="muted")
    console.print(Panel(desc_text, title="[bold accent]DESCRIPTION[/]", border_style="accent", box=box.ROUNDED))
    console.print()
    
    # COMMANDS
    cmd_table = Table(show_header=False, box=None, show_edge=False, padding=(0, 2))
    cmd_table.add_column(style=f"bold {HILITE}", justify="left")
    cmd_table.add_column(style="muted")
    cmd_table.add_row("Lock", "Encrypts and hides a secure payload (secret text or file) inside a cover image using AES-GCM and D-ABC optimization.")
    cmd_table.add_row("Unlock", "Extracts and decrypts a hidden payload from a stego image using D-ABC optimization.")
    cmd_table.add_row("Inspect", "Generates a publication-ready visual analysis (PSNR, SSIM, MSE) comparing cover and stego images.")
    console.print(Panel(cmd_table, title="[bold accent]COMMANDS[/]", border_style="accent", box=box.ROUNDED))
    console.print()
    
    # ADVANCED PARAMETERS
    param_table = Table(show_header=False, box=None, show_edge=False, padding=(0, 2))
    param_table.add_column(style=f"bold {HILITE}", justify="left")
    param_table.add_column(style="muted")
    param_table.add_row("Colony Size", "Number of bees in the D-ABC algorithm (Default: 30). Higher values improve optimization but increase execution time.")
    param_table.add_row("Max Iterations", "Optimization depth for finding optimal pixel coordinates (Default: 50).")
    param_table.add_row("Min Blocks", "Smallest block size for the Adaptive Quadtree (Default: 4). Determines how aggressively the algorithm targets complex/edged areas.")
    console.print(Panel(param_table, title="[bold accent]ADVANCED PARAMETERS[/]", border_style="accent", box=box.ROUNDED))
    console.print()
    
    # EXAMPLES
    ex_text = Text()
    ex_text.append("stegano-product lock -i cover.pgm -o stego.pgm -p \"mypassword\" -m \"secret message\"\n", style="muted")
    ex_text.append("stegano-product unlock -i stego.pgm -o secret.txt -p \"mypassword\" --print", style="muted")
    console.print(Panel(ex_text, title="[bold accent]EXAMPLES[/]", border_style="accent", box=box.ROUNDED))
    console.print()

    # DEVELOPERS & SUPPORT
    dev_table = Table(show_header=False, box=None, show_edge=False, padding=(0, 2))
    dev_table.add_column(style=f"bold {HILITE}", justify="left")
    dev_table.add_column(style="muted", overflow="fold", no_wrap=False)
    dev_table.add_row("Team", "Berkay Bayramoğlu, Betül Feyza Göksu, Gülnur Durukan, İsmail Erol")
    dev_table.add_row("GitHub", "[underline]github.com/Berkaybbayramoglu/Veri-Yap-lar---Sitenografi-[/underline]")
    dev_table.add_row("Contact", "[underline]tr.linkedin.com/in/berkaybayramoglu[/underline]")
    console.print(Panel(dev_table, title="[bold accent]DEVELOPERS & SUPPORT[/]", border_style="accent", box=box.ROUNDED))
    
    console.input("\n[dim]Press Enter to return to the Command Deck...[/dim]")

def run_interactive(console: Console) -> None:
    _require_questionary(console)
    _require_prompt_toolkit(console)
    _print_branding(console)
    
    welcome = "Welcome to STEGANO. Advanced data concealment powered by D-ABC optimization.\n"
    
    import random
    import time
    import sys
    import os
    
    def _key_pressed() -> bool:
        if os.name == 'nt':
            import msvcrt
            if msvcrt.kbhit():
                while msvcrt.kbhit():
                    msvcrt.getch()
                return True
            return False
        else:
            import select
            import termios
            try:
                fd = sys.stdin.fileno()
                old_settings = termios.tcgetattr(fd)
                new_settings = termios.tcgetattr(fd)
                new_settings[3] = new_settings[3] & ~termios.ICANON & ~termios.ECHO
                try:
                    termios.tcsetattr(fd, termios.TCSANOW, new_settings)
                    r, _, _ = select.select([fd], [], [], 0.0)
                    if r:
                        while select.select([fd], [], [], 0.0)[0]:
                            os.read(fd, 1)
                        return True
                finally:
                    termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
            except Exception:
                pass
            return False

    console.print()
    console.print(Text("> ", style=f"bold {HILITE}"), end="")

    for i, char in enumerate(welcome):
        if _key_pressed():
            sys.stdout.write("\r\033[2K\033[1A\033[2K\r")
            sys.stdout.write(f"> {welcome}")
            sys.stdout.flush()
            break

        console.print(char, end="")
        sys.stdout.flush()
    
        if char in ".,!":
            time.sleep(random.uniform(0.1, 0.3))
        elif char == " ":
            time.sleep(random.uniform(0.04, 0.08))
        else:
            time.sleep(random.uniform(0.01, 0.04))
            
    console.print("\n[dim]Press Enter to continue...[/dim]")
    input()
    console.clear()

    import questionary
    import questionary.prompts.select as q_select
    from questionary import Style
    from questionary.prompts.common import InquirerControl

    OriginalInquirerControl = q_select.InquirerControl

    class _MenuControl(InquirerControl):
        def _get_choice_tokens(self):
            tokens = []
            
            for i, choice in enumerate(self.choices):
                is_selected = (i == self.pointed_at)
                
                title = choice.title
                desc = choice.description or ""
                
                if is_selected:
                    tokens.append(("class:pointer", "┃ "))
                    tokens.append(("class:highlighted", f"{title}"))
                    if desc:
                        tokens.append(("", "\n"))
                        tokens.append(("class:pointer", "┃ "))
                        tokens.append(("class:description", f"{desc}"))
                else:
                    tokens.append(("", "  "))
                    tokens.append(("class:text", f"{title}"))
                    if desc:
                        tokens.append(("", "\n"))
                        tokens.append(("", "  "))
                        tokens.append(("class:description", f"{desc}"))
                
                if i < len(self.choices) - 1:
                    tokens.append(("", "\n\n"))
                    
            return tokens


    menu_style = Style(
        [
            ("qmark", f"fg:{HILITE} bold"),
            ("pointer", f"fg:{HILITE} bold"),       
            ("highlighted", f"fg:{ACCENT} bold"),  
            ("text", "fg:#cccccc"),                
            ("description", "dim"),                
            ("bottom-toolbar", "#b0b0b0 bg:#1a1a1a noreverse"),
        ]
    )

    choices = [
        questionary.Choice(
            "Lock",
            value="lock",
            description="Encrypts and embeds a payload (message or file) into a cover image.",
        ),
        questionary.Choice(
            "Unlock",
            value="unlock",
            description="Extracts and decrypts a hidden payload from a stego image.",
        ),
        questionary.Choice(
            "Inspect",
            value="inspect",
            description="Generates a visual analysis comparing cover and stego images.",
        ),
        questionary.Choice(
            "Help",
            value="help",
            description="Show the help menu.",
        ),
        questionary.Choice(
            "Exit",
            value="exit",
            description="Close the console.",
        ),
    ]

    while True:
        try:
            q_select.InquirerControl = _MenuControl

            prompt_obj = questionary.select(
                "Command deck",
                choices=choices,
                style=menu_style,
                qmark="",
                pointer="", 
                instruction="",
                bottom_toolbar=_menu_toolbar(),
            )

            q_select.InquirerControl = OriginalInquirerControl

            bindings = prompt_obj.application.key_bindings
            
            @bindings.add("q")
            def _q1(event):
                event.app.exit(result="exit")
                
            @bindings.add("?")
            def _help1(event):
                event.app.exit(result="help")
                
            action = prompt_obj.ask()
            
            if action is None:
                return
            if action in {"exit", "quit"}:
                return
            if action == "help":
                _show_help(console)
                continue

            if action == "lock":
                ns = _lock_menu(console)
                cmd_lock(ns, console)
                console.input("\n[dim]Press Enter to return to the Command Deck...[/dim]")
            elif action == "unlock":
                ns = _execute_unlock_protocol(console)
                cmd_unlock(ns, console)
                console.input("\n[dim]Press Enter to return to the Command Deck...[/dim]")
            elif action == "inspect":
                ns = _wizard_inspect(console)
                cmd_inspect(ns, console)
                console.input("\n[dim]Press Enter to return to the Command Deck...[/dim]")

        except KeyboardInterrupt:
            console.print("\n[warn]Cancelled by user.[/warn]\n")
            time.sleep(1.5)
        except (FileNotFoundError, FileExistsError, ValueError, OSError) as exc:
            console.print(f"[error]Error:[/error] {exc}\n")
            console.input("\n[dim]Press Enter to return to the Command Deck...[/dim]")
        except Exception as exc:
            console.print(f"[error]Unexpected error:[/error] {exc}\n")
            console.input("\n[dim]Press Enter to return to the Command Deck...[/dim]")

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="stegano-product",
        description="Product CLI for deterministic steganography",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    sub = p.add_subparsers(dest="command", required=False)

    lock = sub.add_parser("lock", help="Encrypts and embeds a payload (message or file) into a cover image.")
    lock.add_argument("-i", "--input", required=True, help="Cover image")
    lock.add_argument("-o", "--output", required=True, help="Output stego image")
    lock.add_argument("-p", "--password", required=True, help="Password")
    lock.add_argument("-m", "--message", help="Message text to embed")
    lock.add_argument("-f", "--file", help="File path to embed")
    lock.add_argument("--colony-size", type=int, default=30)
    lock.add_argument("--max-iter", type=int, default=50)
    lock.add_argument("--min-block", type=int, default=4)
    lock.add_argument("--force", action="store_true", help="Overwrite outputs")
    lock.set_defaults(func=cmd_lock)

    unlock = sub.add_parser("unlock", help="Extracts and decrypts a hidden payload from a stego image.")
    unlock.add_argument("-i", "--input", required=True, help="Stego image")
    unlock.add_argument("-o", "--output", required=True, help="Output file path")
    unlock.add_argument("-p", "--password", required=True, help="Password")
    unlock.add_argument("--colony-size", type=int, default=30)
    unlock.add_argument("--max-iter", type=int, default=50)
    unlock.add_argument("--min-block", type=int, default=4)
    unlock.add_argument("--force", action="store_true", help="Overwrite outputs")
    unlock.add_argument("--print", action="store_true", help="Attempt to print the extracted payload to the terminal if it is readable text.")
    unlock.set_defaults(func=cmd_unlock)

    insp = sub.add_parser("inspect", help="Generates a visual analysis comparing cover and stego images.")
    insp.add_argument("--cover", required=True, help="Original cover image")
    insp.add_argument("--stego", required=True, help="Stego image")
    insp.add_argument("--output", default="inspection_result.png")
    insp.add_argument("--force", action="store_true", help="Overwrite outputs")
    insp.set_defaults(func=cmd_inspect)

    return p


def main() -> int:
    parser = build_parser()
    ns = parser.parse_args()
    console = Console(theme=THEME)
    if ns.command is None:
        try:
            run_interactive(console)
            return 0
        except KeyboardInterrupt:
            console.print("\n[yellow]Cancelled by user.[/yellow]")
            return 130
        except ModuleNotFoundError:
            return 1
    try:
        return ns.func(ns, console)
    except KeyboardInterrupt:
        console.print("\n[yellow]Cancelled by user.[/yellow]")
        return 130
    except (FileNotFoundError, FileExistsError, ValueError, OSError) as exc:
        console.print(f"[red]Error:[/red] {exc}")
        return 1
    except Exception as exc:
        console.print(f"[red]Unexpected error:[/red] {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
