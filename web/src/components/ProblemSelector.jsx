import React from "react";
import SearchableSelect from "./SearchableSelect";

function displayId(pid) {
  const s = String(pid);
  if (s.length > 12) return s.slice(0, 8);
  return `P${s}`;
}

export default function ProblemSelector({ problems, selected, onChange }) {
  const options = problems.map((p) => {
    const id = displayId(p.problem_id);
    const desc = p.short_prompt
      ? `${id} ${p.short_prompt}`
      : `${id} — ${p.n_samples} samples`;
    return {
      value: p.problem_id,
      label: desc,
      accuracy: p.accuracy,
    };
  });

  const handleChange = (v) => {
    onChange(isNaN(Number(v)) ? v : Number(v));
  };

  // Used for BOTH the selected display and the dropdown list
  const renderDropdownOption = (opt) => {
    if (opt.accuracy == null) return opt.label;
    const pct = Math.round(opt.accuracy * 100);
    const color = pct > 0 ? "#2ecc40" : "#ff4136";
    return (
      <span>
        <span style={{
          color, fontWeight: 600, marginRight: 4,
          fontSize: "0.85em", fontFamily: "monospace",
        }}>
          [{pct}%]
        </span>
        {opt.label}
      </span>
    );
  };

  return (
    <SearchableSelect
      label="Problem"
      options={options}
      value={selected}
      onChange={handleChange}
      renderDropdownOption={renderDropdownOption}
    />
  );
}
