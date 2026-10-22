//! Wall-clock helpers. Session time is `u64` milliseconds since a session's `started_at`.

use anyhow::{Result, bail};
use jiff::{RoundMode, SignedDuration, Timestamp, Unit, Zoned, ZonedRound, civil, tz::TimeZone};

pub fn now() -> Zoned {
    truncate_ms(&Timestamp::now().to_zoned(system_tz()))
}

/// The system time zone, or UTC when there is none (no `/etc/localtime` or `TZ`, as in a
/// container or the Nix build sandbox). jiff calls that zone `Etc/Unknown`, which it writes
/// but cannot parse back, so a `meta.toml` saved with it could never be read.
pub fn system_tz() -> TimeZone {
    known_or_utc(TimeZone::system())
}

fn known_or_utc(tz: TimeZone) -> TimeZone {
    if tz.is_unknown() { TimeZone::UTC } else { tz }
}

pub fn truncate_ms(z: &Zoned) -> Zoned {
    z.round(
        ZonedRound::new()
            .smallest(Unit::Millisecond)
            .mode(RoundMode::Trunc),
    )
    .expect("truncating to milliseconds cannot overflow")
}

/// Parses a start time: RFC 9557 (`...+02:00[Europe/Rome]`), RFC 3339 with an offset,
/// or a naive local date-time (`2026-10-08T15:00:12`, `2026-10-08 15:00`), which is
/// interpreted in the system time zone.
pub fn parse_start(s: &str) -> Result<Zoned> {
    let s = s.trim();
    if let Ok(z) = s.parse::<Zoned>() {
        return Ok(truncate_ms(&z));
    }
    if let Ok(t) = s.parse::<Timestamp>() {
        return Ok(truncate_ms(&t.to_zoned(system_tz())));
    }
    if let Ok(dt) = s.parse::<civil::DateTime>() {
        return Ok(truncate_ms(&dt.to_zoned(system_tz())?));
    }
    bail!("cannot read {s:?} as a date and time (example: 2026-10-08T15:00:00)")
}

pub fn add_ms(z: &Zoned, ms: u64) -> Zoned {
    z.checked_add(SignedDuration::from_millis(ms as i64))
        .expect("session offsets fit in jiff's range")
}

/// Milliseconds from `a` to `b`, or 0 if `b` is before `a`.
pub fn ms_between(a: &Zoned, b: &Zoned) -> u64 {
    b.duration_since(a).as_millis().clamp(0, u64::MAX as i128) as u64
}

pub fn format_duration(ms: u64) -> String {
    let mins = ms / 60_000;
    match mins {
        0 => "<1 min".to_string(),
        1..=59 => format!("{mins} min"),
        _ => format!("{} h {} min", mins / 60, mins % 60),
    }
}

/// Session IDs are the local start time: `2026-10-08T090312`.
pub fn session_id(z: &Zoned) -> String {
    z.strftime("%Y-%m-%dT%H%M%S").to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn an_unknown_system_zone_becomes_utc() {
        assert_eq!(known_or_utc(TimeZone::unknown()), TimeZone::UTC);
        let plus_two = TimeZone::fixed(jiff::tz::offset(2));
        assert_eq!(known_or_utc(plus_two.clone()), plus_two);
    }

    #[test]
    fn parse_start_accepts_zoned_offset_and_naive_forms() {
        let z = parse_start("2026-10-08T09:03:12.418+02:00[+02:00]").unwrap();
        assert_eq!(z.time(), jiff::civil::time(9, 3, 12, 418_000_000));

        let z = parse_start("2026-10-08T09:03:12+02:00").unwrap();
        assert_eq!(
            z.timestamp(),
            "2026-10-08T07:03:12Z".parse::<jiff::Timestamp>().unwrap()
        );

        for s in [
            "2026-10-08T15:00:12",
            "2026-10-08 15:00:12",
            " 2026-10-08T15:00:12\n",
        ] {
            let z = parse_start(s).unwrap();
            assert_eq!(z.datetime().to_string(), "2026-10-08T15:00:12", "{s:?}");
        }
        assert!(parse_start("yesterday").is_err());
    }

    #[test]
    fn add_and_measure_ms() {
        let a = parse_start("2026-10-08T09:00:00+02:00[+02:00]").unwrap();
        let b = add_ms(&a, 90_500);
        assert_eq!(
            b,
            "2026-10-08T09:01:30.5+02:00[+02:00]"
                .parse::<Zoned>()
                .unwrap()
        );
        assert_eq!(ms_between(&a, &b), 90_500);
        assert_eq!(ms_between(&b, &a), 0);
    }

    #[test]
    fn format_duration_cases() {
        assert_eq!(format_duration(0), "<1 min");
        assert_eq!(format_duration(59_999), "<1 min");
        assert_eq!(format_duration(60_000), "1 min");
        assert_eq!(format_duration(3_599_999), "59 min");
        assert_eq!(format_duration(31_020_000), "8 h 37 min");
    }

    #[test]
    fn session_id_and_truncation() {
        let z = parse_start("2026-10-08T09:03:12.418999+02:00[+02:00]").unwrap();
        assert_eq!(session_id(&z), "2026-10-08T090312");
        assert_eq!(truncate_ms(&z).subsec_nanosecond(), 418_000_000);
    }
}
