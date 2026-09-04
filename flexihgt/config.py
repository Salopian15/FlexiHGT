"""Config-file support for the command line tools.

Probably something i shouldve done soonah. 
Should be functional.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, Sequence

logger = logging.getLogger(__name__)


class ConfigError(ValueError):
    """Raised when a config file cannot be used."""


def load_config(path: Path) -> Dict[str, Any]:
    """Read a JSON or TOML config file into a flat dict of option -> value.

    Keys may be written as they appear on the command line (``--tax_level``,
    ``tax-level``) or as argparse destinations (``tax_level``); all three
    normalise to the same thing.
    """
    path = Path(path)
    if not path.is_file():
        raise ConfigError(f'Config file not found: {path}')

    text = path.read_text(encoding='utf-8')
    suffix = path.suffix.lower()
    try:
        if suffix in ('.toml', '.tml'):
            data = _load_toml(text, path)
        elif suffix in ('.json', ''):
            data = json.loads(text)
        else:
            raise ConfigError(
                f'Unsupported config format {suffix!r}; use .toml or .json'
            )
    except json.JSONDecodeError as exc:
        raise ConfigError(f'Could not parse {path}: {exc}') from exc

    if not isinstance(data, dict):
        raise ConfigError(f'Config file must contain a table/object: {path}')

    # A [flexihgt] section is honoured so a config can sit alongside others.
    if 'flexihgt' in data and isinstance(data['flexihgt'], dict):
        data = data['flexihgt']

    normalised = {
        str(key).lstrip('-').replace('-', '_'): value
        for key, value in data.items()
    }
    logger.info('Loaded %d setting(s) from %s', len(normalised), path)
    return normalised


def _load_toml(text: str, path: Path) -> Dict[str, Any]:
    try:
        import tomllib                       # Python 3.11+
    except ImportError as exc:               # pragma: no cover - version dependent
        raise ConfigError(
            f'Reading {path} needs TOML support (Python 3.11+). '
            'Use a .json config file instead.'
        ) from exc
    try:
        return tomllib.loads(text)
    except tomllib.TOMLDecodeError as exc:
        raise ConfigError(f'Could not parse {path}: {exc}') from exc


def apply_config(parser, args, config: Dict[str, Any], argv: Sequence[str]) -> None:
    """Fill in ``args`` from ``config`` without overriding explicit flags.

    An option is treated as explicit if it appears in ``argv``; everything else
    is still at its parser default and may be replaced.

    Values go through the same ``type=`` conversion and ``choices=`` check the
    command line would apply.  Assigning them raw let a config file smuggle in
    ``tax_level = "Family"`` or ``evalue = "1e-5"``, which then failed much
    later -- or, for options nothing else validates, not at all.
    """
    explicit = _explicit_destinations(parser, argv)
    actions = {action.dest: action for action in parser._actions}  # noqa: SLF001 - argparse has no public API

    unknown = sorted(set(config) - set(actions))
    if unknown:
        raise ConfigError(
            f'Unknown option(s) in config: {", ".join(unknown)}. '
            f'Valid options: {", ".join(sorted(set(actions) - {"help"}))}'
        )

    for key, value in config.items():
        if key in explicit:
            logger.debug('Command line overrides config for %s', key)
            continue
        setattr(args, key, _validated(actions[key], key, value))


def _validated(action, key: str, value: Any) -> Any:
    """Convert and check one config value the way argparse would."""
    if value is None:
        return None

    flag_action = action.__class__.__name__ in ('_StoreTrueAction', '_StoreFalseAction')
    if flag_action:
        if not isinstance(value, bool):
            raise ConfigError(f'{key} is a flag and needs true or false, not {value!r}')
        return value

    if action.type is not None and not isinstance(value, bool):
        try:
            value = action.type(value)
        except (TypeError, ValueError) as exc:
            raise ConfigError(f'Invalid value for {key}: {value!r} ({exc})') from exc

    if action.choices is not None and value not in action.choices:
        raise ConfigError(
            f'Invalid value for {key}: {value!r}; '
            f'expected one of {", ".join(str(choice) for choice in action.choices)}'
        )
    return value


def _explicit_destinations(parser, argv: Sequence[str]) -> set:
    """Destinations named by a flag actually present on the command line."""
    lookup = {}
    for action in parser._actions:  # noqa: SLF001
        for option in action.option_strings:
            lookup[option] = action.dest

    explicit = set()
    for token in argv:
        if not token.startswith('-'):
            continue
        name = token.split('=', 1)[0]
        if name in lookup:
            explicit.add(lookup[name])
    return explicit
