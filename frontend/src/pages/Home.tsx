import { useAuth } from "../context/AuthContext";

export default function Home() {
  const { user } = useAuth();
  return (
    <div>
      <h2 className="text-2xl font-semibold mb-2">Welcome{user ? `, ${user.email}` : ""}</h2>
      <p className="text-gray-600">{"RAG App"} is running. Generated pages appear in the nav automatically.</p>
    </div>
  );
}
