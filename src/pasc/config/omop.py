"""Site-level OMOP settings shared by every module.

The thesis analysed AIR·MS, where the OMOP CDM v5.3 tables live in the schema
``CDMPHI`` (Appendix A.4). The modules format that name into their SQL through
``CDM_SCHEMA``; indicator specifications and a few templates spell it out
literally, which pasc.db.connect() rewrites at execution time when the environment
variable ``OMOP_CDM_SCHEMA`` names a different schema.
"""
import os

DEFAULT_CDM_SCHEMA = "CDMPHI"
CDM_SCHEMA = os.getenv("OMOP_CDM_SCHEMA", DEFAULT_CDM_SCHEMA).strip() or DEFAULT_CDM_SCHEMA
