import json

from boamp_pipeline.episodes import IndexedDisjointSet
from boamp_pipeline.standardize import standardize_record


def test_nested_raw_values_remain_valid_json() -> None:
    record = {
        "idweb": "24-1",
        "dateparution": "2024-01-01",
        "donnees": {"nested": [1, {"label": "marché"}]},
    }

    standardized = standardize_record(record, "fixture.jsonl")

    assert json.loads(standardized["donnees"]) == record["donnees"]
    assert standardized["date_outside_study"] is False


def test_eforms_list_valued_legal_entities_and_addresses() -> None:
    record = {
        "idweb": "24-2",
        "dateparution": "2024-02-01",
        "nomacheteur": "Acheteur test",
        "donnees": {
            "EFORMS": {
                "ContractNotice": {
                    "cac:ContractingParty": {
                        "cac:Party": {"cac:PartyIdentification": {"cbc:ID": "ORG-1"}}
                    },
                    "ext:UBLExtensions": {
                        "ext:UBLExtension": {
                            "efac:Organization": {
                                "efac:Company": [
                                    {
                                        "cac:PartyIdentification": {"cbc:ID": "ORG-1"},
                                        "cac:PartyLegalEntity": [
                                            {"cbc:CompanyID": "26440013600471"}
                                        ],
                                        "cac:PostalAddress": [
                                            {"cbc:PostalZone": "44000", "cbc:CityName": "Nantes"}
                                        ],
                                    }
                                ]
                            }
                        }
                    },
                }
            }
        },
    }

    standardized = standardize_record(record, "fixture.jsonl")

    assert standardized["buyer_siret"] == "26440013600471"
    assert standardized["buyer_siren"] == "264400136"
    assert standardized["buyer_department"] == "44"
    assert standardized["buyer_region"] == "Pays de la Loire"
    assert standardized["grand_ouest_flag"] is True


def test_transitive_trusted_buyer_conflict_is_rejected() -> None:
    dsu = IndexedDisjointSet(3)
    dsu.set_trusted_buyer(0, "111111111")
    dsu.set_trusted_buyer(2, "222222222")

    accepted, _ = dsu.union(0, 1)
    conflict_accepted, reason = dsu.union(1, 2)

    assert accepted is True
    assert conflict_accepted is False
    assert reason == "trusted_buyer_conflict"


def test_eforms_reference_uses_direct_contract_folder_only() -> None:
    record = {
        "idweb": "24-3",
        "dateparution": "2024-03-01",
        "donnees": {
            "EFORMS": {
                "ContractNotice": {
                    "cbc:ContractFolderID": "ABC-2024-001",
                    "cac:ProcurementProject": {
                        "cbc:ID": "ORG-0001",
                        "cbc:Description": "DCE",
                    },
                }
            }
        },
    }

    standardized = standardize_record(record, "fixture.jsonl")

    assert standardized["procedure_reference"] == "abc 2024 001"
    assert standardized["procedure_reference_source"] == "eforms:cbc:ContractFolderID"
