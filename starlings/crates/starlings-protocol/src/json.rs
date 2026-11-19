use serde::de::{self, DeserializeSeed, MapAccess, SeqAccess, Visitor};
use serde_json::{Map, Number, Value};
use std::fmt;

pub const MAX_JSON_BYTES: usize = 1_048_576;
const MAX_NODES: usize = 16_384;
const MAX_DEPTH: usize = 32;

// Serde parses JSON syntax; this visitor adds duplicate-key and allocation budgets.
struct Seed<'a> {
    remaining: &'a mut usize,
    depth: usize,
}
impl<'de> DeserializeSeed<'de> for Seed<'_> {
    type Value = Value;
    fn deserialize<D: de::Deserializer<'de>>(self, deserializer: D) -> Result<Value, D::Error> {
        if *self.remaining == 0 || self.depth > MAX_DEPTH {
            return Err(de::Error::custom("JSON budget exceeded"));
        }
        *self.remaining -= 1;
        deserializer.deserialize_any(self)
    }
}
impl<'de> Visitor<'de> for Seed<'_> {
    type Value = Value;
    fn expecting(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("bounded JSON")
    }
    fn visit_bool<E: de::Error>(self, v: bool) -> Result<Value, E> {
        Ok(Value::Bool(v))
    }
    fn visit_i64<E: de::Error>(self, v: i64) -> Result<Value, E> {
        Ok(Value::Number(v.into()))
    }
    fn visit_u64<E: de::Error>(self, v: u64) -> Result<Value, E> {
        Ok(Value::Number(v.into()))
    }
    fn visit_f64<E: de::Error>(self, v: f64) -> Result<Value, E> {
        Number::from_f64(v)
            .map(Value::Number)
            .ok_or_else(|| E::custom("nonfinite number"))
    }
    fn visit_str<E: de::Error>(self, v: &str) -> Result<Value, E> {
        Ok(Value::String(v.to_owned()))
    }
    fn visit_string<E: de::Error>(self, v: String) -> Result<Value, E> {
        Ok(Value::String(v))
    }
    fn visit_unit<E: de::Error>(self) -> Result<Value, E> {
        Ok(Value::Null)
    }
    fn visit_seq<A: SeqAccess<'de>>(self, mut seq: A) -> Result<Value, A::Error> {
        let mut out = Vec::new();
        while let Some(v) = seq.next_element_seed(Seed {
            remaining: self.remaining,
            depth: self.depth + 1,
        })? {
            out.push(v);
        }
        Ok(Value::Array(out))
    }
    fn visit_map<A: MapAccess<'de>>(self, mut map: A) -> Result<Value, A::Error> {
        let mut out = Map::new();
        while let Some(key) = map.next_key::<String>()? {
            if out.contains_key(&key) {
                return Err(de::Error::custom("duplicate JSON key"));
            }
            let value = map.next_value_seed(Seed {
                remaining: self.remaining,
                depth: self.depth + 1,
            })?;
            out.insert(key, value);
        }
        Ok(Value::Object(out))
    }
}

pub fn parse_json(bytes: &[u8]) -> Result<Value, crate::Error> {
    if bytes.len() > MAX_JSON_BYTES {
        return Err(crate::Error("JSON body exceeds limit"));
    }
    let mut deserializer = serde_json::Deserializer::from_slice(bytes);
    let mut remaining = MAX_NODES;
    let value = Seed {
        remaining: &mut remaining,
        depth: 0,
    }
    .deserialize(&mut deserializer)
    .map_err(|_| crate::Error("invalid, duplicate or over-budget JSON"))?;
    deserializer
        .end()
        .map_err(|_| crate::Error("trailing JSON data"))?;
    Ok(value)
}
