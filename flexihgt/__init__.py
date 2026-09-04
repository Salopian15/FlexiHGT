"""FlexiHGT: flexible-taxonomic-level horizontal gene transfer detection."""

from .core import AnalysisResult, HGTDetect, HGTParameters, HGTScores
from .taxonomy import TAX_RANKS, TaxonomyIndex, TaxonomyInfo, TaxonomyProvider

__version__ = '0.2.0'

__all__ = [
    'AnalysisResult',
    'HGTDetect',
    'HGTParameters',
    'HGTScores',
    'TAX_RANKS',
    'TaxonomyIndex',
    'TaxonomyInfo',
    'TaxonomyProvider',
    '__version__',
]
