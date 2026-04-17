import React, { useState, useRef, useEffect, useCallback } from "react";
import styles from "./SearchableSelect.module.css";

export default function SearchableSelect({ label, options, value, onChange, renderOption, renderDropdownOption }) {
  const [open, setOpen] = useState(false);
  const [query, setQuery] = useState("");
  const [focusIdx, setFocusIdx] = useState(-1);
  const wrapperRef = useRef(null);
  const inputRef = useRef(null);

  const filtered = options.filter((opt) => {
    const text = String(opt.label ?? opt);
    return text.toLowerCase().includes(query.toLowerCase());
  });

  const selectedOpt = options.find((o) => (o.value ?? o) === value);

  useEffect(() => {
    const handler = (e) => {
      if (wrapperRef.current && !wrapperRef.current.contains(e.target)) {
        setOpen(false);
        setQuery("");
      }
    };
    document.addEventListener("mousedown", handler);
    return () => document.removeEventListener("mousedown", handler);
  }, []);

  // Focus input when dropdown opens
  useEffect(() => {
    if (open && inputRef.current) {
      inputRef.current.focus();
    }
  }, [open]);

  const handleKeyDown = useCallback((e) => {
    if (!open) {
      if (e.key === "ArrowDown" || e.key === "Enter") {
        setOpen(true);
        e.preventDefault();
      }
      return;
    }
    if (e.key === "ArrowDown") {
      setFocusIdx((i) => Math.min(i + 1, filtered.length - 1));
      e.preventDefault();
    } else if (e.key === "ArrowUp") {
      setFocusIdx((i) => Math.max(i - 1, 0));
      e.preventDefault();
    } else if (e.key === "Enter" && focusIdx >= 0 && focusIdx < filtered.length) {
      const opt = filtered[focusIdx];
      onChange(opt.value ?? opt);
      setOpen(false);
      setQuery("");
      e.preventDefault();
    } else if (e.key === "Escape") {
      setOpen(false);
      setQuery("");
      e.preventDefault();
    }
  }, [open, filtered, focusIdx, onChange]);

  // Render the selected item display (supports JSX via renderDropdownOption)
  const renderSelected = () => {
    if (!selectedOpt) return String(value ?? "");
    const renderer = renderDropdownOption || renderOption;
    if (renderer) return renderer(selectedOpt);
    return selectedOpt.label ?? String(selectedOpt);
  };

  return (
    <div className={styles.wrapper} ref={wrapperRef}>
      {label && <label className={styles.label}>{label}</label>}
      <div className={styles.inputWrapper}>
        {open ? (
          <input
            ref={inputRef}
            className={styles.input}
            value={query}
            onChange={(e) => { setQuery(e.target.value); setFocusIdx(0); }}
            onKeyDown={handleKeyDown}
            placeholder="Search..."
            aria-label={label}
            aria-expanded={true}
            role="combobox"
            aria-haspopup="listbox"
          />
        ) : (
          <div
            className={styles.input}
            onClick={() => { setOpen(true); setQuery(""); }}
            tabIndex={0}
            onKeyDown={handleKeyDown}
            role="combobox"
            aria-expanded={false}
            aria-haspopup="listbox"
            aria-label={label}
          >
            {renderSelected()}
          </div>
        )}
        <span className={styles.chevron} onClick={() => { setOpen(!open); setQuery(""); }}>
          {open ? "\u25B2" : "\u25BC"}
        </span>
      </div>
      {open && (
        <div className={styles.dropdown} role="listbox">
          {filtered.length === 0 && <div className={styles.noResults}>No matches</div>}
          {filtered.map((opt, i) => {
            const optValue = opt.value ?? opt;
            const optDisplay = renderDropdownOption
              ? renderDropdownOption(opt)
              : renderOption
                ? renderOption(opt)
                : (opt.label ?? String(opt));
            return (
              <div
                key={optValue}
                className={`${styles.option}${i === focusIdx ? ` ${styles.focused}` : ""}${optValue === value ? ` ${styles.selected}` : ""}`}
                onClick={() => { onChange(optValue); setOpen(false); setQuery(""); }}
                role="option"
                aria-selected={optValue === value}
              >
                {optDisplay}
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}
