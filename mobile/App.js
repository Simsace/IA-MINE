import Feather from "@expo/vector-icons/Feather";
import * as Clipboard from "expo-clipboard";
import * as Speech from "expo-speech";
import { StatusBar } from "expo-status-bar";
import { useEffect, useMemo, useRef, useState } from "react";
import {
  ActivityIndicator,
  Animated,
  KeyboardAvoidingView,
  Platform,
  Pressable,
  SafeAreaView,
  ScrollView,
  Share,
  StyleSheet,
  Text,
  TextInput,
  useWindowDimensions,
  View
} from "react-native";

const DEFAULT_API_URL = Platform.OS === "web" ? "http://localhost:8001" : "http://192.168.1.15:8001";

const STARTERS = [
  "Comment obtenir un permis de petite mine au Mali ?",
  "Quelles obligations environnementales pour exploiter une mine ?",
  "Quelle est la contribution du secteur extractif au budget ?",
  "Quelles sont les mines d'or au Mali ?"
];

export default function App() {
  const { width } = useWindowDimensions();
  const isWide = width >= 820;
  const scrollRef = useRef(null);
  const fade = useRef(new Animated.Value(0)).current;
  const pulse = useRef(new Animated.Value(0)).current;

  const [apiUrl, setApiUrl] = useState(DEFAULT_API_URL);
  const [sidebarOpen, setSidebarOpen] = useState(isWide);
  const [panel, setPanel] = useState("chat");
  const [mode, setMode] = useState("instant");
  const [question, setQuestion] = useState("");
  const [status, setStatus] = useState("Ready");
  const [loading, setLoading] = useState(false);
  const [history, setHistory] = useState([]);
  const [favorites, setFavorites] = useState([]);
  const [messages, setMessages] = useState([]);

  const canSend = question.trim().length > 0 && !loading;
  const isEmpty = panel === "chat" && messages.length === 0;

  useEffect(() => {
    Animated.timing(fade, { toValue: 1, duration: 360, useNativeDriver: true }).start();
  }, [fade]);

  useEffect(() => {
    if (!loading) {
      pulse.stopAnimation();
      pulse.setValue(0);
      return;
    }
    Animated.loop(
      Animated.sequence([
        Animated.timing(pulse, { toValue: 1, duration: 520, useNativeDriver: true }),
        Animated.timing(pulse, { toValue: 0, duration: 520, useNativeDriver: true })
      ])
    ).start();
  }, [loading, pulse]);

  function scrollEnd() {
    requestAnimationFrame(() => scrollRef.current?.scrollToEnd({ animated: true }));
  }

  function enrichPrompt(text) {
    if (mode === "expert") {
      return `${text}\n\nReponds comme un expert du secteur minier malien: structure, nuance, limites, sans exposer les documents internes.`;
    }
    return `${text}\n\nReponds rapidement, clairement, sans exposer les documents internes.`;
  }

  async function ask(text = question) {
    const clean = text.trim();
    if (!clean || loading) return;

    setPanel("chat");
    setQuestion("");
    setLoading(true);
    setStatus("Thinking");
    setMessages((items) => [...items, { id: `u-${Date.now()}`, role: "user", text: clean }]);
    scrollEnd();

    try {
      const response = await fetch(`${apiUrl.replace(/\/$/, "")}/ask`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question: enrichPrompt(clean) })
      });
      if (!response.ok) throw new Error(`API ${response.status}`);
      const data = await response.json();
      const answer = {
        id: `a-${Date.now()}`,
        role: "assistant",
        text: data.answer,
        confidence: data.confidence,
        engine: data.engine,
        evidenceCount: data.evidence_count || 0
      };
      setMessages((items) => [...items, answer]);
      setHistory((items) => [{ question: clean, answer: data.answer, date: "Today" }, ...items].slice(0, 30));
      setStatus(data.engine === "openai" ? "Advanced" : "Local");
    } catch (error) {
      setMessages((items) => [
        ...items,
        { id: `e-${Date.now()}`, role: "assistant", text: "Connexion impossible. Verifie l'adresse API dans les reglages.", confidence: "faible" }
      ]);
      setStatus(error.message);
    } finally {
      setLoading(false);
      scrollEnd();
    }
  }

  function newChat() {
    setPanel("chat");
    setMessages([]);
    setQuestion("");
    setStatus("Ready");
  }

  async function copy(text) {
    await Clipboard.setStringAsync(text);
    setStatus("Copied");
  }

  async function share(text) {
    try {
      await Share.share({ message: text });
    } catch {
      setStatus("Share unavailable");
    }
  }

  function speak(text) {
    Speech.stop();
    Speech.speak(text.slice(0, 1600), { language: "fr-FR", rate: 0.96 });
  }

  function favorite(message) {
    setFavorites((items) => (items.some((item) => item.id === message.id) ? items : [{ ...message, date: "Today" }, ...items]));
    setStatus("Saved");
  }

  function regenerate() {
    const last = [...messages].reverse().find((item) => item.role === "user");
    if (last) ask(last.text);
  }

  const appear = {
    opacity: fade,
    transform: [{ translateY: fade.interpolate({ inputRange: [0, 1], outputRange: [10, 0] }) }]
  };

  return (
    <SafeAreaView style={styles.safe}>
      <StatusBar style="dark" />
      <KeyboardAvoidingView style={styles.root} behavior={Platform.OS === "ios" ? "padding" : undefined}>
        <View style={styles.app}>
          {(sidebarOpen || isWide) && (
            <Sidebar
              isWide={isWide}
              panel={panel}
              setPanel={setPanel}
              newChat={newChat}
              history={history}
              favorites={favorites}
              close={() => setSidebarOpen(false)}
            />
          )}

          <View style={styles.main}>
            <View style={styles.topBar}>
              <Pressable style={styles.topIcon} onPress={() => setSidebarOpen((value) => !value)}>
                <Feather name="sidebar" size={20} color="#7b8491" />
              </Pressable>
              <Text style={styles.topTitle}>MineIntel Mali</Text>
              <Pressable style={styles.topIcon} onPress={() => setPanel("settings")}>
                <Feather name="settings" size={20} color="#7b8491" />
              </Pressable>
            </View>

            <Animated.View style={[styles.content, appear]}>
              {panel === "chat" && (
                <>
                  {isEmpty ? (
                    <EmptyState mode={mode} setMode={setMode} ask={ask} />
                  ) : (
                    <ScrollView ref={scrollRef} style={styles.thread} contentContainerStyle={styles.threadContent} keyboardShouldPersistTaps="handled">
                      {messages.map((message) => (
                        <Message
                          key={message.id}
                          message={message}
                          copy={copy}
                          share={share}
                          speak={speak}
                          favorite={favorite}
                          regenerate={regenerate}
                        />
                      ))}
                      {loading && <Thinking pulse={pulse} />}
                    </ScrollView>
                  )}

                  <Composer
                    question={question}
                    setQuestion={setQuestion}
                    canSend={canSend}
                    ask={ask}
                    mode={mode}
                    setMode={setMode}
                    compact={!isEmpty}
                  />
                </>
              )}

              {panel === "history" && <List title="Historique" items={history} empty="Aucune conversation pour le moment." />}
              {panel === "favorites" && <List title="Favoris" items={favorites} empty="Aucune reponse favorite." />}
              {panel === "settings" && <Settings apiUrl={apiUrl} setApiUrl={setApiUrl} status={status} />}
            </Animated.View>
          </View>
        </View>
      </KeyboardAvoidingView>
    </SafeAreaView>
  );
}

function Mark({ small = false }) {
  return (
    <View style={[styles.mark, small && styles.markSmall]}>
      <View style={[styles.markPeak, small && styles.markPeakSmall]} />
      <View style={[styles.markSpark, small && styles.markSparkSmall]} />
    </View>
  );
}

function Sidebar({ isWide, panel, setPanel, newChat, history, favorites, close }) {
  const today = history.slice(0, 4);
  const older = history.slice(4, 10);

  function go(nextPanel) {
    setPanel(nextPanel);
    if (!isWide) close();
  }

  return (
    <View style={[styles.sidebar, !isWide && styles.sidebarMobile]}>
      <View style={styles.sideBrand}>
        <View style={styles.brandRow}>
          <Mark />
          <Text style={styles.brandText}>MineIntel</Text>
        </View>
        <Pressable style={styles.collapseButton} onPress={close}>
          <Feather name="sidebar" size={18} color="#8a94a3" />
        </Pressable>
      </View>

      <Pressable style={styles.newChat} onPress={newChat}>
        <Feather name="plus-circle" size={18} color="#111827" />
        <Text style={styles.newChatText}>New chat</Text>
      </Pressable>

      <ScrollView style={styles.historyList} showsVerticalScrollIndicator={false}>
        <NavItem icon="message-square" label="Discussion" active={panel === "chat"} onPress={() => go("chat")} />
        <NavItem icon="clock" label={`Historique (${history.length})`} active={panel === "history"} onPress={() => go("history")} />
        <NavItem icon="star" label={`Favoris (${favorites.length})`} active={panel === "favorites"} onPress={() => go("favorites")} />

        <Text style={styles.groupTitle}>Today</Text>
        {today.length ? today.map((item, index) => <HistoryItem key={`t-${index}`} text={item.question} onPress={() => go("history")} />) : <Text style={styles.emptySide}>Aucune analyse</Text>}

        <Text style={styles.groupTitle}>Recent</Text>
        {older.length ? older.map((item, index) => <HistoryItem key={`o-${index}`} text={item.question} onPress={() => go("history")} />) : <Text style={styles.emptySide}>Les conversations recentes apparaitront ici</Text>}
      </ScrollView>

      <View style={styles.profile}>
        <View style={styles.avatar}>
          <Text style={styles.avatarText}>M</Text>
        </View>
        <Text style={styles.profileText}>MineIntel Mali</Text>
        <Feather name="more-horizontal" size={18} color="#8a94a3" />
      </View>
    </View>
  );
}

function NavItem({ icon, label, active, onPress }) {
  return (
    <Pressable style={[styles.navItem, active && styles.navActive]} onPress={onPress}>
      <Feather name={icon} size={16} color={active ? "#2f64ff" : "#64748b"} />
      <Text style={[styles.navText, active && styles.navTextActive]} numberOfLines={1}>{label}</Text>
    </Pressable>
  );
}

function HistoryItem({ text, onPress }) {
  return (
    <Pressable style={styles.historyItem} onPress={onPress}>
      <Text style={styles.historyText} numberOfLines={1}>{text}</Text>
    </Pressable>
  );
}

function EmptyState({ mode, setMode, ask }) {
  return (
    <View style={styles.emptyCenter}>
      <View style={styles.emptyTitleRow}>
        <Mark small />
        <Text style={styles.emptyTitle}>Start chatting with MineIntel</Text>
      </View>

      <View style={styles.modeSwitch}>
        <Pressable style={[styles.modePill, mode === "instant" && styles.modePillActive]} onPress={() => setMode("instant")}>
          <Feather name="zap" size={15} color={mode === "instant" ? "#2f64ff" : "#111827"} />
          <Text style={[styles.modeText, mode === "instant" && styles.modeTextActive]}>Instant</Text>
        </Pressable>
        <Pressable style={[styles.modePill, mode === "expert" && styles.modePillActive]} onPress={() => setMode("expert")}>
          <Feather name="shield" size={15} color={mode === "expert" ? "#2f64ff" : "#111827"} />
          <Text style={[styles.modeText, mode === "expert" && styles.modeTextActive]}>Expert</Text>
        </Pressable>
      </View>

      <View style={styles.suggestions}>
        {STARTERS.map((item) => (
          <Pressable key={item} style={styles.suggestionChip} onPress={() => ask(item)}>
            <Text style={styles.suggestionText} numberOfLines={1}>{item}</Text>
          </Pressable>
        ))}
      </View>
    </View>
  );
}

function Composer({ question, setQuestion, canSend, ask, mode, setMode, compact }) {
  return (
    <View style={[styles.composerShell, compact && styles.composerShellCompact]}>
      <View style={[styles.composer, compact && styles.composerCompact]}>
        <TextInput
          value={question}
          onChangeText={setQuestion}
          multiline
          style={styles.input}
          placeholder="Message MineIntel"
          placeholderTextColor="#a5afbd"
        />

        <View style={styles.composerBottom}>
          <View style={styles.toolRow}>
            <Pressable style={[styles.toolChip, mode === "expert" && styles.toolChipActive]} onPress={() => setMode(mode === "expert" ? "instant" : "expert")}>
              <Feather name="cpu" size={15} color="#2f64ff" />
              <Text style={styles.toolText}>DeepMine</Text>
            </Pressable>
            <Pressable style={styles.toolChip}>
              <Feather name="globe" size={15} color="#2f64ff" />
              <Text style={styles.toolText}>Search</Text>
            </Pressable>
          </View>

          <View style={styles.composerActions}>
            <Pressable style={styles.attachButton}>
              <Feather name="paperclip" size={19} color="#222936" />
            </Pressable>
            <Pressable style={[styles.sendButton, !canSend && styles.sendDisabled]} disabled={!canSend} onPress={() => ask()}>
              <Feather name="arrow-up" size={22} color="#ffffff" />
            </Pressable>
          </View>
        </View>
      </View>
    </View>
  );
}

function Message({ message, copy, share, speak, favorite, regenerate }) {
  if (message.role === "user") {
    return (
      <View style={styles.userWrap}>
        <View style={styles.userBubble}>
          <Text style={styles.userText}>{message.text}</Text>
        </View>
      </View>
    );
  }

  return (
    <View style={styles.assistantWrap}>
      <Mark small />
      <View style={styles.assistantBody}>
        <Text style={styles.assistantText}>{message.text}</Text>
        <View style={styles.metaRow}>
          <View style={styles.metaBadge}>
            <Feather name="lock" size={12} color="#627189" />
            <Text style={styles.metaText}>Base interne</Text>
          </View>
          {!!message.evidenceCount && (
            <View style={styles.metaBadge}>
              <Feather name="activity" size={12} color="#2f64ff" />
              <Text style={styles.metaText}>Confiance {message.confidence}</Text>
            </View>
          )}
        </View>
        <View style={styles.answerActions}>
          <AnswerAction icon="copy" label="Copier" onPress={() => copy(message.text)} />
          <AnswerAction icon="share-2" label="Partager" onPress={() => share(message.text)} />
          <AnswerAction icon="volume-2" label="Lire" onPress={() => speak(message.text)} />
          <AnswerAction icon="star" label="Favori" onPress={() => favorite(message)} />
          <AnswerAction icon="refresh-cw" label="Refaire" onPress={regenerate} />
        </View>
      </View>
    </View>
  );
}

function AnswerAction({ icon, label, onPress }) {
  return (
    <Pressable style={styles.answerAction} onPress={onPress}>
      <Feather name={icon} size={14} color="#7b8491" />
      <Text style={styles.answerActionText}>{label}</Text>
    </Pressable>
  );
}

function Thinking({ pulse }) {
  const scale = pulse.interpolate({ inputRange: [0, 1], outputRange: [0.72, 1.12] });
  return (
    <View style={styles.assistantWrap}>
      <Mark small />
      <View style={styles.thinking}>
        {[0, 1, 2].map((item) => (
          <Animated.View key={item} style={[styles.thinkingDot, { transform: [{ scale }] }]} />
        ))}
        <ActivityIndicator color="#2f64ff" />
      </View>
    </View>
  );
}

function List({ title, items, empty }) {
  return (
    <ScrollView style={styles.panel} contentContainerStyle={styles.panelContent}>
      <Text style={styles.panelTitle}>{title}</Text>
      {!items.length && <Text style={styles.panelEmpty}>{empty}</Text>}
      {items.map((item, index) => (
        <View key={`${title}-${index}`} style={styles.panelCard}>
          <Text style={styles.panelCardTitle}>{item.question || item.text?.slice(0, 90)}</Text>
          <Text style={styles.panelCardText}>{item.answer || item.text}</Text>
          {!!item.date && <Text style={styles.panelDate}>{item.date}</Text>}
        </View>
      ))}
    </ScrollView>
  );
}

function Settings({ apiUrl, setApiUrl, status }) {
  return (
    <ScrollView style={styles.panel} contentContainerStyle={styles.panelContent}>
      <Text style={styles.panelTitle}>Reglages</Text>
      <Text style={styles.settingLabel}>Adresse API</Text>
      <TextInput value={apiUrl} onChangeText={setApiUrl} autoCapitalize="none" autoCorrect={false} style={styles.settingInput} />
      <Text style={styles.settingHint}>Statut: {status}</Text>
      <Text style={styles.settingHint}>Les documents restent cote serveur et ne sont pas visibles par l'utilisateur.</Text>
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  safe: {
    flex: 1,
    backgroundColor: "#ffffff"
  },
  root: {
    flex: 1
  },
  app: {
    flex: 1,
    flexDirection: "row",
    backgroundColor: "#ffffff"
  },
  sidebar: {
    width: 260,
    backgroundColor: "#f7f8fb",
    borderRightWidth: 1,
    borderRightColor: "#edf0f4",
    paddingHorizontal: 12,
    paddingTop: 20,
    paddingBottom: 14
  },
  sidebarMobile: {
    position: "absolute",
    zIndex: 10,
    left: 0,
    top: 0,
    bottom: 0
  },
  sideBrand: {
    height: 42,
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "space-between",
    marginBottom: 22
  },
  brandRow: {
    flexDirection: "row",
    alignItems: "center",
    gap: 8
  },
  brandText: {
    fontSize: 24,
    color: "#2f64ff",
    fontWeight: "800"
  },
  collapseButton: {
    width: 32,
    height: 32,
    borderRadius: 8,
    alignItems: "center",
    justifyContent: "center"
  },
  newChat: {
    height: 44,
    borderRadius: 22,
    backgroundColor: "#ffffff",
    borderWidth: 1,
    borderColor: "#e7eaf0",
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "center",
    gap: 8,
    shadowColor: "#1f2937",
    shadowOpacity: 0.05,
    shadowRadius: 12,
    shadowOffset: { width: 0, height: 4 },
    elevation: 2,
    marginBottom: 18
  },
  newChatText: {
    color: "#111827",
    fontSize: 15,
    fontWeight: "600"
  },
  historyList: {
    flex: 1
  },
  navItem: {
    height: 38,
    borderRadius: 8,
    paddingHorizontal: 10,
    flexDirection: "row",
    alignItems: "center",
    gap: 9,
    marginBottom: 4
  },
  navActive: {
    backgroundColor: "#edf3ff"
  },
  navText: {
    flex: 1,
    color: "#475569",
    fontSize: 14,
    fontWeight: "500"
  },
  navTextActive: {
    color: "#2f64ff",
    fontWeight: "700"
  },
  groupTitle: {
    color: "#8a94a3",
    fontSize: 13,
    fontWeight: "700",
    marginTop: 18,
    marginBottom: 8,
    paddingHorizontal: 10
  },
  historyItem: {
    minHeight: 38,
    justifyContent: "center",
    paddingHorizontal: 10,
    borderRadius: 8
  },
  historyText: {
    color: "#20242b",
    fontSize: 14
  },
  emptySide: {
    color: "#a2abb8",
    fontSize: 13,
    paddingHorizontal: 10,
    lineHeight: 18
  },
  profile: {
    height: 42,
    flexDirection: "row",
    alignItems: "center",
    gap: 10,
    marginTop: 10
  },
  avatar: {
    width: 30,
    height: 30,
    borderRadius: 15,
    alignItems: "center",
    justifyContent: "center",
    backgroundColor: "#ff5a2f"
  },
  avatarText: {
    color: "#ffffff",
    fontWeight: "800"
  },
  profileText: {
    flex: 1,
    color: "#4b5563",
    fontSize: 14,
    fontWeight: "600"
  },
  main: {
    flex: 1,
    backgroundColor: "#ffffff"
  },
  topBar: {
    height: 48,
    paddingHorizontal: 16,
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "space-between"
  },
  topIcon: {
    width: 34,
    height: 34,
    borderRadius: 8,
    alignItems: "center",
    justifyContent: "center"
  },
  topTitle: {
    color: "#9aa3af",
    fontSize: 13,
    fontWeight: "600"
  },
  content: {
    flex: 1
  },
  emptyCenter: {
    flex: 1,
    alignItems: "center",
    justifyContent: "center",
    paddingHorizontal: 20,
    paddingBottom: 160
  },
  emptyTitleRow: {
    flexDirection: "row",
    alignItems: "center",
    gap: 12,
    marginBottom: 24
  },
  emptyTitle: {
    color: "#020617",
    fontSize: 25,
    fontWeight: "800",
    letterSpacing: 0
  },
  modeSwitch: {
    height: 43,
    flexDirection: "row",
    alignItems: "center",
    backgroundColor: "#ffffff",
    borderWidth: 1,
    borderColor: "#e1e5ec",
    borderRadius: 22,
    padding: 2,
    marginBottom: 36
  },
  modePill: {
    height: 39,
    minWidth: 132,
    borderRadius: 20,
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "center",
    gap: 7
  },
  modePillActive: {
    backgroundColor: "#eef4ff",
    borderWidth: 1,
    borderColor: "#b9cdfd"
  },
  modeText: {
    color: "#111827",
    fontSize: 15,
    fontWeight: "600"
  },
  modeTextActive: {
    color: "#2f64ff"
  },
  suggestions: {
    width: "100%",
    maxWidth: 760,
    flexDirection: "row",
    flexWrap: "wrap",
    justifyContent: "center",
    gap: 8,
    marginBottom: 12
  },
  suggestionChip: {
    height: 34,
    maxWidth: 340,
    paddingHorizontal: 13,
    borderRadius: 17,
    backgroundColor: "#f6f8fc",
    borderWidth: 1,
    borderColor: "#edf0f5",
    justifyContent: "center"
  },
  suggestionText: {
    color: "#526071",
    fontSize: 13,
    fontWeight: "500"
  },
  composerShell: {
    position: "absolute",
    left: 0,
    right: 0,
    bottom: 48,
    alignItems: "center",
    paddingHorizontal: 20
  },
  composerShellCompact: {
    position: "relative",
    bottom: 0,
    paddingBottom: 16
  },
  composer: {
    width: "100%",
    maxWidth: 776,
    minHeight: 124,
    borderRadius: 20,
    backgroundColor: "#ffffff",
    borderWidth: 1,
    borderColor: "#e2e6ed",
    shadowColor: "#0f172a",
    shadowOpacity: 0.08,
    shadowRadius: 24,
    shadowOffset: { width: 0, height: 14 },
    elevation: 4,
    paddingHorizontal: 14,
    paddingTop: 14,
    paddingBottom: 12
  },
  composerCompact: {
    minHeight: 96,
    borderRadius: 18
  },
  input: {
    flex: 1,
    minHeight: 46,
    maxHeight: 150,
    color: "#111827",
    fontSize: 16,
    lineHeight: 22,
    paddingHorizontal: 2,
    paddingVertical: 2
  },
  composerBottom: {
    minHeight: 38,
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "space-between",
    gap: 12
  },
  toolRow: {
    flexDirection: "row",
    alignItems: "center",
    gap: 8
  },
  toolChip: {
    height: 34,
    paddingHorizontal: 12,
    borderRadius: 17,
    backgroundColor: "#eef4ff",
    borderWidth: 1,
    borderColor: "#c7d7ff",
    flexDirection: "row",
    alignItems: "center",
    gap: 6
  },
  toolChipActive: {
    backgroundColor: "#e4edff"
  },
  toolText: {
    color: "#2f64ff",
    fontSize: 14,
    fontWeight: "600"
  },
  composerActions: {
    flexDirection: "row",
    alignItems: "center",
    gap: 10
  },
  attachButton: {
    width: 36,
    height: 36,
    borderRadius: 18,
    alignItems: "center",
    justifyContent: "center"
  },
  sendButton: {
    width: 36,
    height: 36,
    borderRadius: 18,
    backgroundColor: "#90a8ff",
    alignItems: "center",
    justifyContent: "center"
  },
  sendDisabled: {
    backgroundColor: "#c9d4ff"
  },
  thread: {
    flex: 1
  },
  threadContent: {
    width: "100%",
    maxWidth: 820,
    alignSelf: "center",
    paddingHorizontal: 20,
    paddingTop: 30,
    paddingBottom: 24,
    gap: 22
  },
  userWrap: {
    alignItems: "flex-end"
  },
  userBubble: {
    maxWidth: "78%",
    backgroundColor: "#f3f5f8",
    borderRadius: 16,
    paddingHorizontal: 15,
    paddingVertical: 11
  },
  userText: {
    color: "#111827",
    fontSize: 15,
    lineHeight: 22
  },
  assistantWrap: {
    flexDirection: "row",
    alignItems: "flex-start",
    gap: 12
  },
  assistantBody: {
    flex: 1,
    paddingTop: 3
  },
  assistantText: {
    color: "#111827",
    fontSize: 16,
    lineHeight: 25
  },
  metaRow: {
    flexDirection: "row",
    flexWrap: "wrap",
    gap: 8,
    marginTop: 12
  },
  metaBadge: {
    minHeight: 26,
    borderRadius: 13,
    paddingHorizontal: 9,
    backgroundColor: "#f6f8fc",
    borderWidth: 1,
    borderColor: "#edf0f5",
    flexDirection: "row",
    alignItems: "center",
    gap: 5
  },
  metaText: {
    color: "#627189",
    fontSize: 12,
    fontWeight: "600"
  },
  answerActions: {
    flexDirection: "row",
    flexWrap: "wrap",
    gap: 8,
    marginTop: 12
  },
  answerAction: {
    minHeight: 28,
    borderRadius: 14,
    paddingHorizontal: 9,
    backgroundColor: "#f7f8fb",
    flexDirection: "row",
    alignItems: "center",
    gap: 5
  },
  answerActionText: {
    color: "#7b8491",
    fontSize: 12,
    fontWeight: "600"
  },
  thinking: {
    height: 38,
    borderRadius: 19,
    paddingHorizontal: 14,
    backgroundColor: "#f6f8fc",
    flexDirection: "row",
    alignItems: "center",
    gap: 8
  },
  thinkingDot: {
    width: 7,
    height: 7,
    borderRadius: 4,
    backgroundColor: "#2f64ff"
  },
  panel: {
    flex: 1
  },
  panelContent: {
    width: "100%",
    maxWidth: 820,
    alignSelf: "center",
    paddingHorizontal: 24,
    paddingTop: 40,
    gap: 12
  },
  panelTitle: {
    color: "#111827",
    fontSize: 28,
    fontWeight: "800",
    marginBottom: 10
  },
  panelEmpty: {
    color: "#8a94a3",
    fontSize: 15
  },
  panelCard: {
    borderRadius: 14,
    backgroundColor: "#ffffff",
    borderWidth: 1,
    borderColor: "#edf0f5",
    padding: 14
  },
  panelCardTitle: {
    color: "#111827",
    fontSize: 15,
    fontWeight: "700",
    marginBottom: 6
  },
  panelCardText: {
    color: "#667085",
    fontSize: 14,
    lineHeight: 21
  },
  panelDate: {
    color: "#a2abb8",
    fontSize: 12,
    marginTop: 8
  },
  settingLabel: {
    color: "#111827",
    fontSize: 14,
    fontWeight: "700",
    marginBottom: 6
  },
  settingInput: {
    height: 48,
    borderRadius: 14,
    borderWidth: 1,
    borderColor: "#e2e6ed",
    paddingHorizontal: 12,
    color: "#111827",
    backgroundColor: "#ffffff",
    marginBottom: 12
  },
  settingHint: {
    color: "#667085",
    fontSize: 14,
    lineHeight: 21
  },
  mark: {
    width: 31,
    height: 31,
    borderRadius: 9,
    backgroundColor: "#2f64ff",
    alignItems: "center",
    justifyContent: "center",
    position: "relative"
  },
  markSmall: {
    width: 27,
    height: 27,
    borderRadius: 8
  },
  markPeak: {
    width: 15,
    height: 15,
    borderLeftWidth: 7.5,
    borderRightWidth: 7.5,
    borderBottomWidth: 15,
    borderLeftColor: "transparent",
    borderRightColor: "transparent",
    borderBottomColor: "#ffffff",
    transform: [{ translateY: -1 }]
  },
  markPeakSmall: {
    width: 13,
    height: 13,
    borderLeftWidth: 6.5,
    borderRightWidth: 6.5,
    borderBottomWidth: 13
  },
  markSpark: {
    position: "absolute",
    right: 7,
    bottom: 7,
    width: 7,
    height: 7,
    borderRadius: 4,
    backgroundColor: "#7ff7e5"
  },
  markSparkSmall: {
    right: 6,
    bottom: 6,
    width: 6,
    height: 6
  }
});
