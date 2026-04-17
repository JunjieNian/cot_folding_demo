import React from "react";
import SearchableSelect from "./SearchableSelect";

export default function SampleSelector({ samples, selected, onChange }) {
  const options = samples.map((s) => {
    const unitLabelPlural = s.unit_label_plural || "slices";
    return {
      value: s.sample_id,
      label: `S${s.sample_id} — ${s.n_slices} ${unitLabelPlural}, ${s.n_transitions} trans`,
      is_correct: s.is_correct,
    };
  });

  // Used for BOTH the selected display and the dropdown list
  const renderDropdownOption = (opt) => {
    if (opt.is_correct == null) return opt.label;
    const symbol = opt.is_correct ? "\u2713" : "\u2717";
    const color = opt.is_correct ? "#2ecc40" : "#ff4136";
    return (
      <span>
        <span style={{ color, fontWeight: 700, marginRight: 4 }}>{symbol}</span>
        {opt.label}
      </span>
    );
  };

  return (
    <SearchableSelect
      label="Sample"
      options={options}
      value={selected}
      onChange={(v) => onChange(Number(v))}
      renderDropdownOption={renderDropdownOption}
    />
  );
}
