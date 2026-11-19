use serde::{Deserialize, Deserializer, Serialize, Serializer, de};
use std::{fmt, str::FromStr};

macro_rules! ids {
    ($($name:ident),+ $(,)?) => {$ (
        #[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
        #[serde(try_from = "String", into = "String")]
        pub struct $name(uuid::Uuid);
        impl TryFrom<String> for $name {
            type Error = &'static str;
            fn try_from(value: String) -> Result<Self, Self::Error> {
                let id = uuid::Uuid::parse_str(&value).map_err(|_| "invalid UUID")?;
                if id.to_string() != value || id.get_version_num() != 4 || id.get_variant() != uuid::Variant::RFC4122 {
                    return Err("expected canonical UUID v4");
                }
                Ok(Self(id))
            }
        }
        impl FromStr for $name {
            type Err = &'static str;
            fn from_str(value: &str) -> Result<Self, Self::Err> { Self::try_from(value.to_owned()) }
        }
        impl From<$name> for String { fn from(value: $name) -> Self { value.0.to_string() } }
        impl fmt::Display for $name {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result { self.0.fmt(f) }
        }
    )+};
}
ids!(
    NodeId,
    ClusterId,
    SnapshotId,
    PlanId,
    LeaseId,
    SessionId,
    GenerationId,
    ClientId,
    InvocationId,
    RequestId
);

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(try_from = "String", into = "String")]
pub struct Digest(String);
impl TryFrom<String> for Digest {
    type Error = &'static str;
    fn try_from(value: String) -> Result<Self, Self::Error> {
        if value.len() != 64
            || !value
                .bytes()
                .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
        {
            return Err("expected lowercase SHA-256");
        }
        Ok(Self(value))
    }
}
impl From<Digest> for String {
    fn from(value: Digest) -> Self {
        value.0
    }
}
impl fmt::Display for Digest {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(f)
    }
}
impl Digest {
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(try_from = "String", into = "String")]
pub struct Decimal(pub u64);
impl TryFrom<String> for Decimal {
    type Error = &'static str;
    fn try_from(value: String) -> Result<Self, Self::Error> {
        let n = value.parse::<u64>().map_err(|_| "invalid u64 decimal")?;
        if n.to_string() != value {
            return Err("noncanonical u64 decimal");
        }
        Ok(Self(n))
    }
}
impl From<Decimal> for String {
    fn from(value: Decimal) -> Self {
        value.0.to_string()
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(try_from = "u32", into = "u32")]
pub struct Version;
impl TryFrom<u32> for Version {
    type Error = &'static str;
    fn try_from(value: u32) -> Result<Self, Self::Error> {
        if value == 1 {
            Ok(Self)
        } else {
            Err("unsupported protocol version")
        }
    }
}
impl From<Version> for u32 {
    fn from(_: Version) -> Self {
        1
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BoundedU32<const MIN: u32, const MAX: u32>(u32);
impl<const MIN: u32, const MAX: u32> BoundedU32<MIN, MAX> {
    pub fn new(value: u32) -> Result<Self, &'static str> {
        if (MIN..=MAX).contains(&value) {
            Ok(Self(value))
        } else {
            Err("integer out of range")
        }
    }
    pub fn get(self) -> u32 {
        self.0
    }
}
impl<const MIN: u32, const MAX: u32> Serialize for BoundedU32<MIN, MAX> {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.serialize_u32(self.0)
    }
}
impl<'de, const MIN: u32, const MAX: u32> Deserialize<'de> for BoundedU32<MIN, MAX> {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        Self::new(u32::deserialize(deserializer)?).map_err(de::Error::custom)
    }
}
